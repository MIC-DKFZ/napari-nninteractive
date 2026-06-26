import contextlib
import os
import warnings
from pathlib import Path
from typing import Any, Optional

import nnInteractive  # lightweight: only reads the package version at import time
import numpy as np
from napari.utils.notifications import show_warning
from napari.viewer import Viewer
from qtpy.QtCore import QEvent
from qtpy.QtWidgets import QApplication, QWidget

# NOTE: torch, nnunetv2 and batchgenerators are only needed for *local* inference
# (the nnInteractive[local] extra). They are imported lazily inside
# _construct_local_session() so a remote-only install (nnInteractive[client]) stays
# PyTorch-free.
from napari_nninteractive.widget_controls import LayerControls

try:
    from nnInteractive.inference.remote import (
        ServerAtCapacityError,
        SessionExpiredError,
    )
except ImportError:  # remote client extra not installed
    class SessionExpiredError(Exception):  # type: ignore[no-redef]
        pass

    class ServerAtCapacityError(Exception):  # type: ignore[no-redef]
        pass


try:
    import httpx

    # A killed or unreachable server surfaces as a transport-level error
    # (connection refused, timeouts, protocol errors) rather than a typed lease
    # error. Treat those the same as an expired session so the Connect button
    # resets and the segmentation is preserved for a reconnect.
    _SESSION_LOST_ERRORS: tuple = (SessionExpiredError, httpx.TransportError)
except ImportError:  # httpx ships with the remote client extra
    _SESSION_LOST_ERRORS = (SessionExpiredError,)


class nnInteractiveWidget(LayerControls):
    """
    A widget for the nnInteractive plugin in Napari that manages model inference sessions
    and allows interactive layer-based actions.

    Handling the in-progress object when a session ends
    ---------------------------------------------------
    Whenever a live session is torn down we have to decide what happens to the
    object the user is currently working on (the un-committed "nnInteractive -
    Label Layer"). The behaviour deliberately splits along *why* the session
    ended:

    * **User-triggered reinitialization** -- changing the model, the Local/Remote
      mode, the server URL/key (``on_model_selected``), the local checkpoint
      (``on_checkpoint_changed``) or a baked-in option such as torch.compile or
      the interaction storage backend (``on_local_settings_changed``). The new
      session cannot meaningfully continue the old object, so we *wrap it up*:
      ``_store_in_progress_segmentation`` commits it as a finished object (exactly
      like "Next Object") and the user starts fresh on the next Initialize. If
      they don't want the stored object they can simply delete the layer.

    * **Unintentional loss** -- the remote lease expired or the connection dropped
      (``_handle_session_expired``). The user did not ask to stop, so we instead
      *resume*: the label layer is kept and the resume machinery
      (``_resume_after_reconnect`` / ``_resume_image_layer`` / ``_resuming``,
      consumed in ``LayerControls.on_init``) seeds the reconnected session with it
      so refinement continues on the same object.

    In short: deliberate resets bank the work and start over; accidental drops
    preserve and resume it.
    """

    def __init__(self, viewer: Viewer, parent: Optional[QWidget] = None):
        """
        Initialize the nnInteractiveWidget.
        """
        # Set before super().__init__ because BaseGUI.__init__ calls _unlock_session,
        # which is overridden below to read self._remote_connected.
        self._remote_connected = False
        super().__init__(viewer, parent)
        self.session = None
        # Resume-after-reconnect state. When a remote session is lost we keep
        # the label layer and, on the next Initialize, seed the new session with
        # it instead of starting from scratch. _resume_image_layer pins the
        # resume to a specific image layer object (identity, not just shape) so a
        # different image with the same shape can never be resumed by mistake.
        self._resume_after_reconnect = False
        self._resume_image_layer = None
        self._resuming = False
        # Checkpoint-path text the current session was built from. Lets a
        # re-submitted, unchanged path be a no-op instead of an uninitialize.
        self._active_checkpoint_text = None
        self._viewer.dims.events.order.connect(self.on_axis_change)

        # Belt-and-suspenders lease release on shutdown. closeEvent on this
        # widget does NOT fire reliably when napari quits (the dock widget
        # tree is destroyed without per-child closeEvent), and the Ctrl+Q
        # path raises SystemExit via quit(), bypassing Qt shutdown entirely.
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._release_session)

        # Catch the napari main window's close at the source via event
        # filter. This is the most reliable hook: it fires synchronously
        # when the user clicks the X, before the dock-widget teardown.
        with contextlib.suppress(Exception):
            qt_window = self._viewer.window._qt_window
            qt_window.installEventFilter(self)
            self._napari_qt_window = qt_window

    def eventFilter(self, obj, event):  # noqa: N802 - Qt API
        if (
            getattr(self, "_napari_qt_window", None) is not None
            and obj is self._napari_qt_window
            and event.type() == QEvent.Close
        ):
            self._release_session()
        return super().eventFilter(obj, event)

    def _unlock_session(self):
        """Same as BaseGUI, but keep Initialize disabled until Connect succeeds in remote mode."""
        super()._unlock_session()
        if self._remote_mode and not self._remote_connected:
            self.init_button.setEnabled(False)

    def _close(self):
        """Ctrl+Q handler: release the lease before quit() raises SystemExit."""
        self._release_session()
        super()._close()

    def closeEvent(self, event):  # noqa: N802 - Qt API
        """Release the remote lease when the widget is being torn down."""
        self._release_session()
        super().closeEvent(event)

    # Event Handlers
    def on_init(self, *args, **kwargs):
        """
        Initialize the inference session and setup layers for interaction.

        In remote mode the session must already be claimed via Connect; this
        method only uploads the image and target buffer. In local mode the
        session is constructed here from the configured checkpoint.
        """
        if self._remote_mode and not self._remote_connected:
            show_warning("Remote mode: please Connect to a server first.")
            return

        super().on_init(*args, **kwargs)

        if self.session is None:
            self._construct_local_session()

        # Shared point for local + remote: surface the model license now (after
        # Initialize) so both modes display it identically.
        self._update_license_display(getattr(self.session, "license", None))

        # Enable only interaction tools supported by the loaded checkpoint.
        supported = self.session.supported_interactions
        self._set_interaction_button_support(
            {
                0: bool(supported.get("points", False)),
                1: bool(supported.get("bbox2d", False)),
                2: bool(supported.get("scribble", False)),
                3: bool(supported.get("lasso", False)),
            }
        )

        _data = self._viewer.layers[self.session_cfg["name"]].data
        _data = _data[np.newaxis, ...]

        if self.source_cfg["ndim"] == 2:
            _data = _data[np.newaxis, ...]

        try:
            self.session.set_image(_data, {"spacing": self.session_cfg["spacing"]})
            self.session.set_target_buffer(self._data_result)
            # Resuming after a reconnect: seed the fresh session with the
            # segmentation we kept so the user continues refining the same
            # object instead of starting over.
            if self._resuming and np.any(self._data_result):
                self.session.add_initial_seg_interaction(
                    (self._data_result > 0).astype(np.uint8), run_prediction=False
                )
        except _SESSION_LOST_ERRORS:
            self._handle_session_expired()
            return

        # Init succeeded; clear the resume state so a normal re-init starts fresh.
        self._resume_after_reconnect = False
        self._resuming = False
        # Remember the checkpoint text this session was built from, so re-pressing
        # Enter on an unchanged path keeps the session instead of resetting it.
        self._active_checkpoint_text = self.model_selection_local.text()

        if self._viewer.dims.not_displayed != ():
            self._scribble_brush_size = self.session.preferred_scribble_thickness[
                self._viewer.dims.not_displayed[0]
            ]
        else:
            self._scribble_brush_size = self.session.preferred_scribble_thickness[
                self._viewer.dims.order[0]
            ]
        # Set the prompt type to positive
        self.prompt_button._uncheck()
        self.prompt_button._check(0)

    def _construct_local_session(self) -> None:
        """Construct the local inference session from self.checkpoint_path."""
        # Heavy, local-only dependencies (the nnInteractive[local] extra). Imported
        # here so remote-only installs never need torch / nnU-Net.
        import torch
        from batchgenerators.utilities.file_and_folder_operations import join, load_json
        from nnunetv2.utilities.find_class_by_name import recursive_find_python_class

        # Get inference class from Checkpoint
        if Path(self.checkpoint_path).joinpath("inference_session_class.json").is_file():
            inference_class = load_json(
                Path(self.checkpoint_path).joinpath("inference_session_class.json")
            )
            if isinstance(inference_class, dict):
                inference_class = inference_class["inference_class"]
        else:
            inference_class = "nnInteractiveInferenceSession"

        inference_class = recursive_find_python_class(
            join(nnInteractive.__path__[0], "inference"),
            inference_class,
            "nnInteractive.inference",
        )

        # CPU Fallback if no Cuda is available
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            show_warning(
                "Cuda is not available. Using CPU instead. This will result in longer runtimes and additionally auto-zoom will be disabled for runtime reasons"
            )
            device = torch.device("cpu")
            self.propagate_ckbx.setChecked(False)

        self.session = inference_class(
            device=device,
            use_torch_compile=self.use_torch_compile_ckbx.isChecked(),
            torch_n_threads=os.cpu_count(),
            verbose=False,
            do_autozoom=self.propagate_ckbx.isChecked(),
            interactions_storage=self.interactions_storage_combo.currentText(),
        )

        self.session.initialize_from_trained_model_folder(
            self.checkpoint_path,
            0,
            "checkpoint_final.pth",
        )

    def _claim_remote_session(self, server_url: str, api_key: Optional[str]):
        """Construct a remote session, mapping errors to user-friendly status text.

        Returns the session on success, or None on any failure (status label
        already updated).
        """
        try:
            import httpx
            from nnInteractive.inference.remote import nnInteractiveRemoteInferenceSession
        except ImportError:
            self.remote_status_label.setText(
                "Remote mode requires the client extra: pip install 'nnInteractive[client]'"
            )
            return None

        try:
            session = nnInteractiveRemoteInferenceSession(
                server_url=server_url, api_key=api_key
            )
        except ServerAtCapacityError:
            self.remote_status_label.setText("Server full; try again later.")
            return None
        # Connectivity problems must be handled BEFORE the session-lost case below:
        # httpx.ConnectError/ConnectTimeout are subclasses of httpx.TransportError, so a
        # broad TransportError catch would otherwise swallow them and report the wrong cause.
        except httpx.ConnectError:
            # DNS failure, connection refused, no route — nothing is listening/reachable.
            self.remote_status_label.setText("Cannot reach server; check URL/port.")
            return None
        except httpx.ConnectTimeout:
            self.remote_status_label.setText("Connection timed out; check URL/network.")
            return None
        except httpx.TimeoutException:
            # Connected, but the server did not answer the claim in time.
            self.remote_status_label.setText("Server not responding; try again.")
            return None
        except SessionExpiredError:
            # The connection worked but the server refused/expired the claim itself.
            self.remote_status_label.setText("Claim rejected; try again.")
            return None
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                self.remote_status_label.setText("Invalid API key.")
            elif "text/html" in e.response.headers.get("content-type", ""):
                self.remote_status_label.setText("Not an nnInteractive server (proxy?).")
            else:
                self.remote_status_label.setText(f"Server error {e.response.status_code}.")
            return None
        except httpx.TransportError:
            # Any other network-level failure (proxy, protocol, broken connection).
            self.remote_status_label.setText("Network error; check connection.")
            return None
        except Exception as e:  # noqa: BLE001
            self.remote_status_label.setText(f"Error: {e}")
            return None

        # Honor the current auto-zoom checkbox on the remote session too.
        with contextlib.suppress(Exception):
            session.set_do_autozoom(self.propagate_ckbx.isChecked())

        return session

    def on_connect_toggle(self) -> None:
        """Claim a remote session, or release the held one if already connected."""
        if self._remote_connected:
            self._disconnect_remote()
            # Treat as if the model was reset: drop layers, regrey interactions.
            self._clear_layers()
            self._unlock_session()
            return

        server_url = self.server_url_edit.text().strip()
        if not server_url:
            self.remote_status_label.setText("enter a server URL")
            return

        api_key = self.api_key_edit.text() or None
        session = self._claim_remote_session(server_url, api_key)
        if session is None:
            return

        self.session = session
        self._remote_connected = True
        self.connect_btn.setText("✓ Connected")
        self.remote_status_label.setText(f"connected ({server_url})")
        # Connecting unlocks the Initialize button (via the override).
        self._unlock_session()

    def _release_session(self) -> None:
        """Best-effort lease release. Idempotent and safe to call during shutdown
        (does not touch Qt widgets, since they may already be torn down).

        Only remote sessions hold a server-side lease and expose close();
        local sessions have nothing to release."""
        close = getattr(self.session, "close", None)
        if close is not None:
            try:
                close()
            except Exception as e:  # noqa: BLE001
                # Don't swallow silently: shutdown bugs are otherwise invisible.
                print(f"[napari-nninteractive] lease release failed: {e!r}")
        self.session = None
        self._remote_connected = False

    def _disconnect_remote(self) -> None:
        """Release the lease and reset connection state. Idempotent."""
        self._release_session()
        self.connect_btn.setText("Connect")
        self.remote_status_label.setText("not connected")
        self._update_license_display(None)

    def _handle_session_expired(self) -> None:
        """Server-side lease is gone. Keep the label layer so the user can
        Connect again and resume refining: the next Initialize will seed the new
        session with the surviving segmentation instead of discarding it."""
        show_warning(
            "Server session lost. Reconnect and re-initialize to continue "
            "refining your segmentation."
        )
        self._resume_after_reconnect = True
        self._disconnect_remote()
        self.remote_status_label.setText("session lost")
        self._clear_layers()
        self._unlock_session()

    def on_remote_settings_changed(self, *args, **kwargs) -> None:
        """User edited the URL or API key; invalidate any existing session."""
        if self._remote_connected:
            # The held lease is for the old URL; release it before resetting.
            self._disconnect_remote()
        else:
            self.remote_status_label.setText("not connected")
        # Defer to on_model_selected to clear layers + session and re-lock UI.
        self.on_model_selected()

    def on_mode_switched(self, *args, **kwargs) -> None:
        """Toggle between Local and Remote inference modes."""
        # Remote-only install: local inference is not installed. Snap the switch
        # back to Remote and explain how to enable local instead of entering an
        # unusable Local mode. _uncheck/_check don't re-emit, so no recursion.
        if not self._local_available and self.mode_switch.index == 0:
            self.mode_switch._uncheck()
            self.mode_switch._check(1)
            self._grey_local_switch_button()  # _uncheck cleared the greyed style
            self._show_local_unavailable_dialog()
            return
        if self._remote_connected:
            self._disconnect_remote()
        self._remote_mode = self.mode_switch.index == 1
        self.local_container.setVisible(not self._remote_mode)
        self.remote_container.setVisible(self._remote_mode)
        # Any toggle resets the switch button styles; restore the greyed Local look.
        if not self._local_available:
            self._grey_local_switch_button()
        self.on_model_selected()

    def _store_in_progress_segmentation(self) -> None:
        """Before a genuine reset (model / mode / server change) drops the session,
        store the object currently being worked on.

        A genuine reset starts a fresh session, so the in-progress segmentation
        cannot be resumed (unlike a reconnect or a baked-in option change). Rather
        than silently discarding it, store it as a finished object - exactly like
        'Next Object' does. The user can delete the stored object manually if they
        do not want it. The working layer is then removed: its data has already
        been copied into the stored object, and removing it prevents the same
        object being stored twice if the user resets again before re-initializing.

        Does nothing when there is no non-empty in-progress segmentation.
        """
        if (
            self.session_cfg is None
            or self.label_layer_name not in self._viewer.layers
            or not np.any(self._viewer.layers[self.label_layer_name].data)
        ):
            return

        self._store_current_object()
        self._viewer.layers.remove(self.label_layer_name)

    def on_model_selected(self):
        """Reset the current session completely"""
        # A genuine reset cannot resume the in-progress object, so store it as a
        # finished object before the session is gone instead of losing it.
        self._store_in_progress_segmentation()
        super().on_model_selected()
        self.session = None
        # Genuine reset: the previous model's license no longer applies.
        self._update_license_display(None)
        # A model/mode/server change is a genuine reset, not a reconnect:
        # don't resume the previous segmentation. (on_mode_switched and
        # on_remote_settings_changed both funnel through here.)
        self._resume_after_reconnect = False
        self._resume_image_layer = None

    def _uninitialize_storing_segmentation(self) -> bool:
        """Drop the live session, first storing the in-progress object as a
        finished object (like a model/mode change) instead of resuming it on the
        next Initialize. Returns True if a session was actually torn down, False
        when nothing was initialized.
        """
        if self.session is None:
            # Nothing initialized yet, so there is nothing to store or tear down;
            # the new value is simply picked up at the next Initialize.
            return False
        self._store_in_progress_segmentation()
        self.session = None
        self._clear_layers()
        self._unlock_session()
        return True

    def on_local_settings_changed(self, *args, **kwargs):
        """A baked-in local option (torch.compile / interaction storage) changed.

        The live session was built with the old value, so drop it and force a
        re-Initialize. The new session cannot resume the in-progress object, so
        store it as a finished object first instead of discarding it. The model is
        unchanged, so the displayed license still applies.
        """
        self._uninitialize_storing_segmentation()

    def on_checkpoint_changed(self, *args, **kwargs):
        """The local checkpoint path was edited or cleared (the 'x' button).

        Like a settings change, drop the session and store the in-progress object
        as a finished object before it is lost. The checkpoint may point at a
        different model though, so drop the displayed license; on_init repopulates
        it once the new session is up.
        """
        # Re-submitting the same path the live session was built from changes
        # nothing, so leave the session initialized.
        if (
            self.session is not None
            and self.model_selection_local.text() == self._active_checkpoint_text
        ):
            return
        if self._uninitialize_storing_segmentation():
            self._update_license_display(None)

    def on_image_selected(self):
        """Reset the current sessions interaction but keep the session itself"""
        super().on_image_selected()
        if self.session is not None:
            try:
                self.session.reset_interactions()
            except _SESSION_LOST_ERRORS:
                self._handle_session_expired()

    def on_reset_interactions(self):
        """Reset only the current interaction"""
        _ind = self.interaction_button.index
        super().on_reset_interactions()
        if self.session is not None:
            try:
                self.session.reset_interactions()
            except _SESSION_LOST_ERRORS:
                self._handle_session_expired()
                return

        self._viewer.layers[self.label_layer_name].refresh()

        self.interaction_button._check(_ind)
        self.on_interaction_selected()
        # self.prompt_button._uncheck()
        self.prompt_button._on_button_pressed(0)

    def on_next(self):
        """Reset the Interactions of current session"""
        _ind = self.interaction_button.index
        super().on_next()
        if self.session is not None:
            try:
                self.session.reset_interactions()
            except _SESSION_LOST_ERRORS:
                self._handle_session_expired()
                return

        # if (
        #     self.use_init_ckbx.isChecked()
        #     and self.label_for_init.currentText() in self._viewer.layers
        # ):
        #     self.init_with_mask()

        self._viewer.layers[self.label_layer_name].refresh()

        self.interaction_button._check(_ind)
        self.on_interaction_selected()
        self.prompt_button._check(0)

    def on_run(self):
        """Manual Run button: predict against the (possibly remote) session and
        refresh the label layer, treating a lost connection like session expiry."""
        if self.session is not None:
            try:
                self.session._predict()
            except _SESSION_LOST_ERRORS:
                self._handle_session_expired()
                return
            self._viewer.layers[self.label_layer_name].refresh()

    def on_propagate_ckbx(self, *args, **kwargs):
        if self.session is not None:
            try:
                self.session.set_do_autozoom(self.propagate_ckbx.isChecked())
            except _SESSION_LOST_ERRORS:
                self._handle_session_expired()

    def on_axis_change(self, event: Any):
        """Change the brush size of the scribble layer when the axis changes"""
        if self.session is not None:

            if self._viewer.dims.not_displayed != ():
                self._scribble_brush_size = self.session.preferred_scribble_thickness[
                    self._viewer.dims.not_displayed[0]
                ]
            else:
                self._scribble_brush_size = self.session.preferred_scribble_thickness[
                    self._viewer.dims.order[0]
                ]

            if self.scribble_layer_name in self._viewer.layers:
                self._viewer.layers[self.scribble_layer_name].brush_size = self._scribble_brush_size

    # Inference Behaviour
    def _bbox_to_half_open_intervals(self, data: np.ndarray) -> list[list[float]]:
        """Convert a napari rectangle to backend-style half-open intervals."""
        mins = np.min(data, axis=0).astype(float)
        maxs = np.max(data, axis=0).astype(float)

        # BBoxes are interpreted as half-open intervals in nnInteractive.
        # If an axis is collapsed (common for the fixed axis of a 2D view),
        # expand it to one voxel so the interval remains non-empty.
        # The upper bound may exceed image size by 1 (safe with Python slicing).
        collapsed = mins == maxs
        maxs[collapsed] = maxs[collapsed] + 1.0

        return [[mins[i], maxs[i]] for i in range(len(mins))]

    def add_interaction(self):
        _index = self.interaction_button.index
        _layer_name = self.layer_dict.get(_index)
        if (
            _layer_name is not None
            and _layer_name in self._viewer.layers
            and not self._viewer.layers[_layer_name].is_free()
        ):
            data = self._viewer.layers[_layer_name].get_last()

            self._viewer.layers[_layer_name].run()
            # self.inference(_data, _index)

            if data is not None:
                _prompt = self.prompt_button.index == 0
                _auto_run = self.run_ckbx.isChecked()

                try:
                    if _index == 0:
                        self._viewer.layers[self.point_layer_name].refresh(force=True)
                        self.session.add_point_interaction(data, _prompt, _auto_run)
                    elif _index == 1:
                        bbox = self._bbox_to_half_open_intervals(data)
                        self.session.add_bbox_interaction(bbox, _prompt, _auto_run)
                    elif _index == 2:
                        crop_3d, bbox = data
                        self.session.add_scribble_interaction(crop_3d, _prompt, _auto_run, interaction_bbox=bbox)
                    elif _index == 3:
                        crop_3d, bbox = data
                        self.session.add_lasso_interaction(crop_3d, _prompt, _auto_run, interaction_bbox=bbox)
                except _SESSION_LOST_ERRORS:
                    self._handle_session_expired()
                    return

                # Record which layer holds this interaction's marker so on_undo can remove it.
                self._interaction_history.append(_layer_name)
                self._viewer.layers[self.label_layer_name].refresh()

    def on_undo(self):
        """Undo the most recent interaction for the current object.

        Reverts the segmentation via the backend's single-level undo and removes the visual
        marker of the undone interaction. Only the most recent interaction can be undone; the
        backend re-arms so the next new interaction becomes undoable again.
        """
        if self.session is None:
            return
        if not getattr(self.session, "supports_undo", False):
            show_warning("Undo is not supported by this server. Please update nninteractive-server.")
            return
        try:
            undone = self.session.undo()
        except _SESSION_LOST_ERRORS:
            self._handle_session_expired()
            return

        if not undone:
            show_warning("Nothing to undo.")
            return

        # Remove the visual marker of the undone interaction, if we tracked one.
        if self._interaction_history:
            layer_name = self._interaction_history.pop()
            if layer_name is not None and layer_name in self._viewer.layers:
                layer = self._viewer.layers[layer_name]
                try:
                    layer.remove_last()
                    layer.refresh()
                except Exception as e:  # noqa: BLE001
                    print(f"[napari-nninteractive] could not remove last interaction marker: {e!r}")

        if self.label_layer_name in self._viewer.layers:
            self._viewer.layers[self.label_layer_name].refresh()

    def on_load_mask(self):

        _layer_data = self._viewer.layers[self.label_for_init.currentText()].data

        assert (
            _layer_data.shape == self.session_cfg["shape"]
        )  # Labels and Image should have same shape

        data = _layer_data == self.class_for_init.value()

        if np.any(data):
            if self.session is not None:
                try:
                    self.session.add_initial_seg_interaction(
                        data.astype(np.uint8), run_prediction=self.auto_refine.isChecked()
                    )
                except _SESSION_LOST_ERRORS:
                    self._handle_session_expired()
                    return
                # Undoable via the backend; there is no interaction-layer marker to remove.
                self._interaction_history.append(None)
                self._viewer.layers[self.label_layer_name].refresh()
        else:
            warnings.warn("Mask is not valid - probably its empty", UserWarning, stacklevel=1)
