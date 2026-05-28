import contextlib
import os
import warnings
from pathlib import Path
from typing import Any, Optional

import nnInteractive
import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import join, load_json
from napari.utils.notifications import show_warning
from napari.viewer import Viewer
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from qtpy.QtCore import QEvent
from qtpy.QtWidgets import QApplication, QWidget

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


class nnInteractiveWidget(LayerControls):
    """
    A widget for the nnInteractive plugin in Napari that manages model inference sessions
    and allows interactive layer-based actions.
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
        except SessionExpiredError:
            self._handle_session_expired()
            return

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
            use_torch_compile=False,
            torch_n_threads=os.cpu_count(),
            verbose=False,
            do_autozoom=self.propagate_ckbx.isChecked(),
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
            self.remote_status_label.setText("server is full, try again later")
            return None
        except SessionExpiredError:
            # Extremely unlikely at /claim time, but handle for completeness.
            self.remote_status_label.setText("server rejected the claim; try again")
            return None
        except httpx.ConnectError:
            self.remote_status_label.setText(f"cannot reach {server_url}")
            return None
        except httpx.ConnectTimeout:
            self.remote_status_label.setText(f"timed out reaching {server_url}")
            return None
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                self.remote_status_label.setText("server rejected the API key")
            elif "text/html" in e.response.headers.get("content-type", ""):
                self.remote_status_label.setText(
                    "server returned HTML (HTTP proxy?); set NO_PROXY"
                )
            else:
                self.remote_status_label.setText(
                    f"server error {e.response.status_code}"
                )
            return None
        except Exception as e:  # noqa: BLE001
            self.remote_status_label.setText(f"error: {e}")
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

    def _handle_session_expired(self) -> None:
        """Server-side lease is gone. Reset UI to 'needs Connect + Initialize'."""
        show_warning(
            "Server session expired. Please Connect again and re-initialize."
        )
        self._disconnect_remote()
        self.remote_status_label.setText("session expired")
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
        if self._remote_connected:
            self._disconnect_remote()
        self._remote_mode = self.mode_switch.index == 1
        self.local_container.setVisible(not self._remote_mode)
        self.remote_container.setVisible(self._remote_mode)
        self.on_model_selected()

    def on_model_selected(self):
        """Reset the current session completely"""
        super().on_model_selected()
        self.session = None

    def on_image_selected(self):
        """Reset the current sessions interaction but keep the session itself"""
        super().on_image_selected()
        if self.session is not None:
            try:
                self.session.reset_interactions()
            except SessionExpiredError:
                self._handle_session_expired()

    def on_reset_interactions(self):
        """Reset only the current interaction"""
        _ind = self.interaction_button.index
        super().on_reset_interactions()
        if self.session is not None:
            try:
                self.session.reset_interactions()
            except SessionExpiredError:
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
            except SessionExpiredError:
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

    def on_propagate_ckbx(self, *args, **kwargs):
        if self.session is not None:
            try:
                self.session.set_do_autozoom(self.propagate_ckbx.isChecked())
            except SessionExpiredError:
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
                except SessionExpiredError:
                    self._handle_session_expired()
                    return

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
                except SessionExpiredError:
                    self._handle_session_expired()
                    return
                self._viewer.layers[self.label_layer_name].refresh()
        else:
            warnings.warn("Mask is not valid - probably its empty", UserWarning, stacklevel=1)
