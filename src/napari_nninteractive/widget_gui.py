import importlib.util
import warnings
from typing import Optional

from napari.layers import Image, Labels
from napari.viewer import Viewer
from napari_toolkit.containers import (
    setup_tabwidget,
    setup_vcollapsiblegroupbox,
    setup_vgroupbox,
    setup_vscrollarea,
)
from napari_toolkit.widgets import (
    setup_acknowledgements,
    setup_checkbox,
    setup_combobox,
    setup_hswitch,
    setup_iconbutton,
    setup_label,
    setup_layerselect,
    setup_lineedit,
    setup_pushbutton,
    setup_spinbox,
    setup_vswitch,
)
from napari_toolkit.widgets.buttons.icon_button import setup_icon
from qtpy.QtCore import QSettings, Qt
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QShortcut,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from napari_nninteractive._version_check import VersionChecker, _is_outdated

# Shared wording for the Local tooltip, the start-up notice, and the dialog shown
# when the user clicks the (greyed) Local switch. Kept as a reason + a how-to so the
# dialog can present them as headline + details without repeating itself; the tooltip
# and console print use the two joined. The lightweight ``nninteractive-client``
# distribution is remote-only and torch-free; local inference lives in the full
# ``nnInteractive`` package.
_REMOTE_ONLY_REASON = (
    "Local inference is unavailable: this is a remote-only (client-only) install "
    "without the local nnInteractive backend, so only a remote nninteractive-server "
    "can be used."
)
_ENABLE_LOCAL_STEPS = (
    "To enable local inference (needs an Nvidia GPU):\n"
    "  1. Install PyTorch yourself — it is NOT installed automatically. Pick the build "
    "matching your GPU/CUDA from https://pytorch.org/get-started/locally/\n"
    '  2. pip install "napari-nninteractive[local]"\n'
    "then restart napari. See the README for details."
)
_REMOTE_ONLY_HINT = f"{_REMOTE_ONLY_REASON}\n{_ENABLE_LOCAL_STEPS}"


def _local_inference_available() -> bool:
    """Return True if local inference is installed.

    The lightweight ``nninteractive-client`` ships only the remote client; the
    local engine and its torch / nnU-Net stack live in the full ``nnInteractive``
    package. We probe for the local inference module *without* importing torch:
    ``find_spec`` only locates the module. In a client-only environment the full
    package's last-resort meta-path finder raises ``ModuleNotFoundError`` here
    (see ``nnInteractive.inference.remote._full_required``), which we treat as
    "not available".
    """
    try:
        return importlib.util.find_spec("nnInteractive.inference.inference_session") is not None
    except ModuleNotFoundError:
        return False
    except Exception:  # noqa: BLE001 - never block the GUI on a capability probe
        return False


class BaseGUI(QWidget):
    """
    A base GUI class for building the Base GUI and connect the components with the correct functions.

    Args:
        viewer (Viewer): The Napari viewer instance to connect with the GUI.
        parent (Optional[QWidget], optional): The parent widget. Defaults to None.
    """

    # Label aggregation (Interact > Label Aggregation). Output mode + overlap rule are read
    # at commit time by LayerControls._store_current_object, which switches on these strings.
    OUT_SEPARATE = "Separate layers"
    OUT_INSTANCE = "Instance map"
    OUT_SEMANTIC = "Semantic map (fixed ID)"
    OVERLAP_KEEP = "keep existing"
    OVERLAP_OVERWRITE = "overwrite"

    def __init__(self, viewer: Viewer, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._width = 300
        self.setMinimumWidth(self._width)
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self._viewer = viewer
        self.session_cfg = None
        # Whether a session is currently initialized (locked). Drives the Initialize
        # button's toggle behaviour: initialize when idle, uninitialize when live.
        self._session_locked = False
        # True when the live local session fell back to the CPU (no usable CUDA GPU);
        # drives the persistent red warning shown below Initialize.
        self._local_running_on_cpu = False
        # Whether local inference is installed. A remote-only
        # 'nninteractive-client' install cannot run it, so we start in Remote
        # mode and disable the Local controls (see _init_model_selection).
        self._local_available = _local_inference_available()
        self._remote_mode = not self._local_available
        self._settings = QSettings("MIC-DKFZ", "napari-nninteractive")

        # Be transparent on start-up: tell remote-only users why local inference
        # is off and how to enable it, so a missing Local button is never a mystery.
        if not self._local_available:
            print(f"[napari-nninteractive] {_REMOTE_ONLY_HINT}")

        _main_layout = QVBoxLayout()
        self.setLayout(_main_layout)

        # The GUI is split across two tabs to keep the day-to-day segmentation
        # workflow uncluttered: "Interact" holds everything used while segmenting,
        # "Settings" holds set-once configuration (model, inference options, label
        # aggregation). Each tab scrolls independently so neither overflows the
        # (narrow, possibly short) napari dock.
        _interact_page = QWidget()
        _interact_outer = QVBoxLayout(_interact_page)
        _interact_outer.setContentsMargins(0, 0, 0, 0)
        _, _interact_layout = setup_vscrollarea(_interact_outer)
        _interact_layout.addWidget(self._init_image_selection())  # Image Selection
        _interact_layout.addWidget(self._init_control_buttons())  # Init / Undo / Reset / Next
        _interact_layout.addWidget(self._init_init_buttons())  # Initialize with Segmentation
        _interact_layout.addWidget(self._init_prompt_selection())  # Prompt Selection
        _interact_layout.addWidget(self._init_interaction_selection())  # Interaction Tools
        _interact_layout.addWidget(self._init_label_aggregation())  # Label Aggregation
        _interact_layout.addWidget(self._init_export_button())  # Export
        _ = setup_acknowledgements(_interact_layout, width=self._width)  # Acknowledgements

        _settings_page = QWidget()
        _settings_outer = QVBoxLayout(_settings_page)
        _settings_outer.setContentsMargins(0, 0, 0, 0)
        _, _settings_layout = setup_vscrollarea(_settings_outer)
        _settings_layout.addWidget(self._init_model_selection())  # Model Selection
        _settings_layout.addWidget(self._init_inference_options())  # Auto-zoom

        _tabs = setup_tabwidget(
            _main_layout,
            widgets=[_interact_page, _settings_page],
            page_names=["Interact", "Settings"],
        )
        # Let the tabs fill the dock vertically (setup_tabwidget defaults to a
        # Fixed vertical policy, which would leave dead space below short tabs).
        _tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Update notice, shown below the tabs so it is visible from either tab
        # (filled in asynchronously once PyPI has been queried; hidden until then).
        self.version_status_label = QLabel("")
        self.version_status_label.setWordWrap(True)
        self.version_status_label.setAlignment(Qt.AlignLeft)
        # Let the user select/copy the update command with the mouse or keyboard.
        self.version_status_label.setTextInteractionFlags(
            Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard
        )
        self.version_status_label.setVisible(False)
        _main_layout.addWidget(self.version_status_label)

        # Keep the Interact-tab hotkeys (P/B/S/L, T, R, M, Ctrl+Z) working regardless of
        # which tab is shown. They are QShortcuts parented to Interact-tab widgets and
        # default to Qt.WindowShortcut, which only fires while the parent widget is visible
        # -- so switching to the Settings tab (hiding those widgets) would otherwise disable
        # them. Promote every shortcut to ApplicationShortcut; focused text fields still
        # consume the keys they accept, so typing in a line edit / spin box is unaffected.
        for _shortcut in self.findChildren(QShortcut):
            _shortcut.setContext(Qt.ApplicationShortcut)

        self._unlock_session()
        self._viewer.bind_key("Ctrl+Q", self._close, overwrite=True)

        # Non-blocking check for newer releases on PyPI. Kept as an attribute so
        # it outlives __init__; the daemon thread it spawns never blocks startup.
        self._version_checker = VersionChecker()
        self._version_checker.finished.connect(self._on_version_check_finished)
        self._version_checker.start()

    # Base Behaviour
    def _close(self):
        """Closes the viewer and quits the application."""
        self._viewer.close()
        quit()

    def _on_version_check_finished(self, results: dict) -> None:
        """Show an up-to-date / update-available notice from the PyPI check.

        `results` maps each package name to an `(installed, latest)` tuple; either
        entry may be None (package not installed or PyPI unreachable). When nothing
        could be compared the label stays hidden rather than showing a false notice.
        """
        outdated = [
            pkg
            for pkg, (installed, latest) in results.items()
            if installed and latest and _is_outdated(installed, latest)
        ]
        checkable = any(installed and latest for installed, latest in results.values())

        if not checkable:
            self.version_status_label.setVisible(False)
            return

        self.version_status_label.setVisible(True)
        if outdated:
            # Name only the package(s) actually behind PyPI, and build the upgrade
            # command from those alone. A package whose installed version is newer
            # than the latest release (e.g. a dev/unreleased build) is never listed,
            # so we don't wrongly tell the user to "update" something already ahead.
            names = ", ".join(outdated)
            self.version_status_label.setText(
                f"Update available for {names}. Please run:\n"
                f"pip install -U {' '.join(outdated)}"
            )
            self.version_status_label.setStyleSheet("color: #e8830c; font-weight: bold;")  # orange
        else:
            self.version_status_label.setText("nnInteractive is up to date")
            self.version_status_label.setStyleSheet("color: #2e9e2e;")  # green

    def _unlock_session(self):
        """Unlocks the session, enabling model and image selection, and initializing controls."""
        self._session_locked = False
        self.init_button.setEnabled(True)
        self._update_init_button_text(initialized=False)
        # No live session -> never show the CPU-fallback warning.
        self._update_device_warning()

        # Reset interaction capabilities until a checkpoint is loaded.
        self._set_interaction_button_support({0: True, 1: True, 2: True, 3: True})

        self.reset_button.setEnabled(False)
        self.prompt_button.setEnabled(False)
        self.interaction_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.reset_interaction_button.setEnabled(False)
        self.undo_button.setEnabled(False)
        self.label_for_init.setEnabled(False)
        self.class_for_init.setEnabled(False)
        self.auto_refine.setEnabled(False)
        self.load_mask_btn.setEnabled(False)

    def _set_interaction_button_support(self, supported: dict[int, bool]) -> None:
        """Enable/disable interaction tool buttons and keep a valid active selection."""
        enabled_indices = []
        for idx, button in enumerate(self.interaction_button.buttons):
            is_enabled = bool(supported.get(idx, True))
            button.setEnabled(is_enabled)
            if is_enabled:
                enabled_indices.append(idx)

        if not enabled_indices:
            return

        if self.interaction_button.index not in enabled_indices:
            self.interaction_button._uncheck()
            self.interaction_button._check(enabled_indices[0])

    def _lock_session(self):
        """Locks the session, disabling model and image selection, and enabling control buttons."""
        self._session_locked = True
        # Kept enabled so it can act as an uninitialize toggle while a session is live.
        self.init_button.setEnabled(True)
        self._update_init_button_text(initialized=True)

        self.reset_button.setEnabled(True)
        self.prompt_button.setEnabled(True)
        self.interaction_button.setEnabled(True)
        self.export_button.setEnabled(True)
        self.reset_interaction_button.setEnabled(True)
        self.undo_button.setEnabled(True)
        self.label_for_init.setEnabled(True)
        self.class_for_init.setEnabled(True)
        self.auto_refine.setEnabled(True)
        self.load_mask_btn.setEnabled(True)

    def _clear_layers(self):
        """Abstract function to clear all needed layers"""

    def _init_model_selection(self) -> QGroupBox:
        """Initializes the model selection as a combo box."""
        _group_box, _layout = setup_vgroupbox(text="Model Selection:")

        # Local | Remote mode switch. Start on Remote when local inference is not
        # installed, since Local cannot be used in that case.
        self.mode_switch = setup_hswitch(
            _layout,
            options=["Local", "Remote"],
            function=self.on_mode_switched,
            default=0 if self._local_available else 1,
            fixed_color="rgb(0,100, 167)",
            tooltips="Run inference locally or on a remote nninteractive-server",
        )

        # Remote-only install: local inference is not installed. Keep the Local
        # button enabled — a *disabled* button swallows clicks and can show no
        # feedback — but selecting it pops an explanatory dialog and snaps the
        # switch back to Remote (handled in on_mode_switched). The tooltip explains
        # why local is unavailable and how to enable it.
        if not self._local_available:
            self.mode_switch.buttons[0].setToolTip(_REMOTE_ONLY_HINT)
            self.mode_switch.buttons[1].setToolTip("Run inference on a remote nninteractive-server")
            self._grey_local_switch_button()

        # --- Local container --- #
        self.local_container = QWidget()
        _local_layout = QVBoxLayout()
        _local_layout.setContentsMargins(0, 0, 0, 0)
        self.local_container.setLayout(_local_layout)
        _layout.addWidget(self.local_container)

        # Populate the model dropdown from the nnInteractive backend manifest — the
        # authoritative list of selectable official models, fetched from Hugging Face
        # (remote-first) with an offline cache fallback. The dropdown shows display
        # names; self._model_ids keeps the matching manifest ids by position. If the
        # list can't be loaded at all, the dropdown is left empty and the user can
        # still point to a local checkpoint below or switch to Remote mode.
        self._model_ids: list[str] = []
        model_display_names: list[str] = []
        default_index = 0
        # The model dropdown only drives local inference (in Remote mode the server
        # picks the model), and model discovery lives in the full package. A
        # remote-only install therefore skips loading the list entirely rather than
        # triggering the full-package-required error.
        if self._local_available:
            try:
                from nnInteractive.model_management import get_default_model_id, list_models

                models = list_models()
                self._model_ids = [m["id"] for m in models]
                model_display_names = [m.get("display_name", m["id"]) for m in models]
                default_id = get_default_model_id()
                if default_id in self._model_ids:
                    default_index = self._model_ids.index(default_id)
            except Exception as exc:  # noqa: BLE001 - never block the GUI on model discovery
                warnings.warn(f"Could not load the nnInteractive model list: {exc}", stacklevel=2)

        self.model_selection = setup_combobox(
            _local_layout, options=model_display_names, function=self.on_model_selected
        )
        if model_display_names:
            # Select the manifest default without firing on_model_selected during init
            # (the handler touches subclass state that isn't built yet).
            self.model_selection.blockSignals(True)
            self.model_selection.setCurrentIndex(default_index)
            self.model_selection.blockSignals(False)

        _boxlayout = QHBoxLayout()
        _local_layout.addLayout(_boxlayout)
        self.model_selection_local = setup_lineedit(
            _boxlayout, placeholder="Use Local Checkpoint...", function=self.on_checkpoint_changed
        )

        def _reset_local_ckpt_lineedit():
            self.model_selection_local.setText("")
            self.on_checkpoint_changed()

        btn = setup_iconbutton(
            _boxlayout, "", "delete_shape", self._viewer.theme, function=_reset_local_ckpt_lineedit
        )
        btn.setFixedWidth(30)

        # --- Advanced (local) options --- #
        # Shown inline now that these live on the roomier Settings tab, instead of
        # being folded away. The chosen values are still persisted via QSettings.
        self.advanced_box, _advanced_layout = setup_vgroupbox(_local_layout, text="Advanced")

        self.use_torch_compile_ckbx = setup_checkbox(
            _advanced_layout,
            "use torch.compile",
            self._settings.value("use_torch_compile", False, type=bool),
            tooltips="If checked: enable torch.compile for local inference. The model is compiled "
            "during Initialize, so initialization takes longer, but every prediction afterwards is faster.",
        )

        _storage_layout = QHBoxLayout()
        _advanced_layout.addLayout(_storage_layout)
        setup_label(_storage_layout, "interaction storage")
        self.interactions_storage_combo = setup_combobox(
            _storage_layout,
            options=["auto", "blosc2", "tensor"],
            tooltips="Storage backend for the interaction tensor (local inference only):\n"
            "• auto: dense tensor for smaller images, blosc2 above ~512x512x512 (default)\n"
            "• blosc2: much less RAM, slightly slower\n"
            "• tensor: much more RAM, slightly faster\n"
            "Pick blosc2 manually if you are short on RAM.",
        )
        saved_storage = self._settings.value("interactions_storage", "auto", type=str)
        _storage_idx = self.interactions_storage_combo.findText(saved_storage)
        if _storage_idx >= 0:
            self.interactions_storage_combo.setCurrentIndex(_storage_idx)

        # --- Remote container --- #
        self.remote_container = QWidget()
        _remote_layout = QVBoxLayout()
        _remote_layout.setContentsMargins(0, 0, 0, 0)
        self.remote_container.setLayout(_remote_layout)
        _layout.addWidget(self.remote_container)

        self.server_url_edit = setup_lineedit(
            _remote_layout,
            placeholder="http://gpu-box:1527",
            function=self.on_remote_settings_changed,
            tooltips="URL of the nninteractive-server, including scheme and port",
        )

        _key_layout = QHBoxLayout()
        _remote_layout.addLayout(_key_layout)
        self.api_key_edit = setup_lineedit(
            _key_layout,
            placeholder="API key (optional)",
            function=self.on_remote_settings_changed,
            tooltips="Bearer token; falls back to NN_INTERACTIVE_API_KEY env var",
        )
        self.api_key_edit.setEchoMode(QLineEdit.Password)

        self.connect_btn = setup_pushbutton(
            _key_layout,
            "Connect",
            function=self.on_connect_toggle,
            tooltips="Optional: test the connection to the nninteractive-server. "
            "Initialize connects automatically if you skip this.",
        )
        self.connect_btn.setFixedWidth(110)

        self.remote_status_label = QLabel("not connected")
        self.remote_status_label.setWordWrap(True)
        _remote_layout.addWidget(self.remote_status_label)

        # Show the container for the active mode. A remote-only install starts in
        # Remote mode; its Local controls are hidden and disabled (the menus that
        # belong to the now-greyed Local button).
        self.local_container.setVisible(not self._remote_mode)
        self.remote_container.setVisible(self._remote_mode)
        if not self._local_available:
            self.local_container.setEnabled(False)

        # Restore last-used values (blocking signals so we don't trigger
        # on_model_selected / on_remote_settings_changed before the rest of
        # the GUI has been built).
        saved_local = self._settings.value("local_checkpoint", "", type=str)
        if saved_local:
            self.model_selection_local.blockSignals(True)
            self.model_selection_local.setText(saved_local)
            self.model_selection_local.blockSignals(False)

        saved_url = self._settings.value("server_url", "", type=str)
        if saved_url:
            self.server_url_edit.blockSignals(True)
            self.server_url_edit.setText(saved_url)
            self.server_url_edit.blockSignals(False)

        # Persist on every edit. API key is intentionally NOT persisted.
        self.model_selection_local.textChanged.connect(
            lambda t: self._settings.setValue("local_checkpoint", t)
        )
        self.server_url_edit.textChanged.connect(lambda t: self._settings.setValue("server_url", t))

        # Persist the advanced option values between sessions.
        self.use_torch_compile_ckbx.toggled.connect(
            lambda checked: self._settings.setValue("use_torch_compile", checked)
        )
        self.interactions_storage_combo.currentTextChanged.connect(
            lambda t: self._settings.setValue("interactions_storage", t)
        )

        # torch.compile and interaction storage are baked into the session at Initialize.
        # Changing one afterwards would leave the GUI out of sync with the live session, so
        # uninitialize and force a re-Initialize -- but keep the in-progress segmentation.
        # Wired after the construction-time restore above, so it never fires during build.
        self.use_torch_compile_ckbx.toggled.connect(lambda *_: self.on_local_settings_changed())
        self.interactions_storage_combo.currentTextChanged.connect(
            lambda *_: self.on_local_settings_changed()
        )

        _group_box.setLayout(_layout)
        return _group_box

    def _grey_local_switch_button(self) -> None:
        """Lightly grey the Local switch button so it reads as unavailable while
        staying clickable (clicking it explains how to enable local inference).

        The switch resets each button's stylesheet whenever it toggles, so this is
        re-applied after switch changes (see on_mode_switched).
        """
        self.mode_switch.buttons[0].setStyleSheet("color: gray;")

    def _show_local_unavailable_dialog(self) -> None:
        """Explain why Local is unavailable and how to enable local inference.

        Shown when the user clicks the greyed (but still clickable) Local switch.
        Deliberately offers no one-click install: the correct PyTorch build depends
        on the user's GPU/CUDA and must be installed manually first (see the steps).
        """
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Information)
        box.setWindowTitle("Local inference not installed")
        box.setText(_REMOTE_ONLY_REASON)
        box.setInformativeText(_ENABLE_LOCAL_STEPS)
        box.addButton(QMessageBox.Close)
        box.setDefaultButton(QMessageBox.Close)
        box.exec()

    def _init_image_selection(self) -> QGroupBox:
        """Initializes the image selection combo box in a group box."""
        _group_box, _layout = setup_vgroupbox(text="Image Selection:")

        self.image_selection = setup_layerselect(
            _layout, viewer=self._viewer, layer_type=Image, function=self.on_image_selected
        )

        _group_box.setLayout(_layout)
        return _group_box

    def _init_control_buttons(self) -> QGroupBox:
        """Initializes the control buttons (Initialize and Reset)."""
        _group_box, _layout = setup_vgroupbox(text="")

        self.init_button = setup_iconbutton(
            _layout,
            "Initialize",
            "new_labels",
            self._viewer.theme,
            self.on_init_button,
            tooltips="Initialize the Model and Image Pair",
        )

        # License of the loaded model, shown directly below Initialize once a
        # session is ready (set in on_init for both local and remote modes).
        self.model_license_label = QLabel("")
        self.model_license_label.setWordWrap(True)
        _layout.addWidget(self.model_license_label)

        # Persistent CPU-fallback warning, shown below Initialize only after a local
        # session was built on the CPU (no usable CUDA GPU). CPU inference is very slow
        # and a transient notification is easy to miss, so this stays visible until the
        # next (successful) Initialize. Hidden by default; toggled by _update_device_warning().
        self.device_warning_label = QLabel("")
        self.device_warning_label.setWordWrap(True)
        self.device_warning_label.setTextFormat(Qt.RichText)
        self.device_warning_label.setOpenExternalLinks(True)
        self.device_warning_label.setStyleSheet("color: #d9534f; font-weight: bold; padding: 4px;")
        self.device_warning_label.setVisible(False)
        _layout.addWidget(self.device_warning_label)

        self.undo_button = setup_iconbutton(
            _layout,
            "Undo (Ctrl+Z)",
            "step_left",
            self._viewer.theme,
            self.on_undo,
            tooltips="Undo the last interaction for the current object - press Ctrl+Z",
            shortcut="Ctrl+Z",
        )
        self.reset_interaction_button = setup_iconbutton(
            _layout,
            "Reset Object (R)",
            "delete",
            self._viewer.theme,
            self.on_reset_interactions,
            tooltips="Keep Model and Image Pair, just reset the interactions for the current object  - press R",
            shortcut="R",
        )
        self.reset_button = setup_iconbutton(
            _layout,
            "Next Object (M)",
            "step_right",
            self._viewer.theme,
            self.on_next,
            tooltips="Keep current segmentation and go to the next object - press M",
            shortcut="M",
        )

        _group_box.setLayout(_layout)
        return _group_box

    def _update_init_button_text(self, initialized: bool) -> None:
        """Reflect the active inference mode on the Initialize button.

        The button reads e.g. ``Initialize (local)`` before a session exists and
        ``Initialized (local)`` once one is live, swapping ``local``/``remote`` with
        the current mode. Called from _lock_session / _unlock_session (which run on
        construction, on every reset, and on mode switch), so the label always
        tracks the real state without a dedicated handler.
        """
        mode = "remote" if self._remote_mode else "local"
        label = "Initialized" if initialized else "Initialize"
        self.init_button.setText(f"{label} ({mode})")
        self.init_button.setToolTip(
            "Uninitialize: tear down the current session (your objects are kept)"
            if initialized
            else "Initialize the Model and Image Pair"
        )

    def _update_license_display(self, license_str: Optional[str]) -> None:
        """Show the loaded model's license below the Initialize button.

        Pass None to clear it (session reset / disconnect). The "!!MISSING!!"
        sentinel is shown as a warning. license_str is the short identifier from
        the checkpoint's LICENSE file (its first line).
        """
        label = self.model_license_label
        if not license_str:
            label.setText("")
            label.setStyleSheet("")
            return
        if license_str.strip() == "!!MISSING!!":
            label.setText("Model license: UNKNOWN (warning!)")
            label.setStyleSheet("color: #d9534f; font-weight: bold;")  # warning red
            return
        label.setText(f"Model license: {license_str.strip()}")
        label.setStyleSheet("")

    def _update_device_warning(self) -> None:
        """Show or hide the persistent CPU-fallback warning below Initialize.

        Visible only while a local session that fell back to the CPU is live (no
        usable CUDA GPU). Hidden for remote and GPU sessions, and whenever nothing is
        initialized. CPU inference is very slow, so -- unlike the transient
        notification -- this stays put until the next (successful) Initialize.
        """
        show = self._session_locked and not self._remote_mode and self._local_running_on_cpu
        if show:
            url = (
                "https://github.com/MIC-DKFZ/napari-nninteractive"
                "#step-3--optional-enable-local-inference"
            )
            self.device_warning_label.setText(
                "⚠ No GPU detected — nnInteractive is running on the CPU, which is very "
                "slow. Likely either no compatible GPU is present, or the installed "
                "PyTorch build does not match your GPU. See the "
                f'<a href="{url}">installation instructions</a> for installing a CUDA '
                "build of PyTorch."
            )
        self.device_warning_label.setVisible(show)

    def _init_init_buttons(self):
        """Initializes the control buttons (Initialize and Reset)."""
        _group_box, _layout = setup_vcollapsiblegroupbox(
            text="Initialize with Segmentation:", collapsed=True
        )

        h_layout = QHBoxLayout()

        self.label_for_init = setup_layerselect(
            h_layout, viewer=self._viewer, layer_type=Labels, stretch=4
        )

        _text = setup_label(h_layout, "Class ID:", stretch=2)
        _text.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        _text.setFixedWidth(70)
        self.class_for_init = setup_spinbox(h_layout, maximum=999, default=1, stretch=1)
        self.class_for_init.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Minimum)

        _layout.addLayout(h_layout)

        self.load_mask_btn = setup_iconbutton(
            _layout,
            "Initialize with Mask",
            "logo_silhouette",
            self._viewer.theme,
            self.on_load_mask,
        )

        self.auto_refine = setup_checkbox(
            _layout, "Auto refine", False, tooltips="Auto Refine the Initial Mask"
        )

        _txt = setup_label(
            _layout, "<b>Warning:</b> This will reset all interactions<br>for the current object"
        )
        _group_box.setLayout(_layout)

        _group_box.setLayout(_layout)
        return _group_box

    def _init_prompt_selection(self) -> QGroupBox:
        """Initializes the prompt selection as switch with options and shortcuts."""
        # The toggle shortcut (T) flips the whole switch rather than picking one
        # option, so it belongs on the section header, not the individual buttons.
        _group_box, _layout = setup_vgroupbox(text="Prompt Type (T):")

        self.prompt_button = setup_hswitch(
            _layout,
            options=["positive", "negative"],
            function=self.on_prompt_selected,
            default=0,
            fixed_color="rgb(0,100, 167)",
            shortcut="T",
            tooltips="Press T to switch",
        )

        _group_box.setLayout(_layout)
        return _group_box

    def _init_interaction_selection(self) -> QGroupBox:
        """Initializes the interaction selection as switch with options and shortcuts."""
        _group_box, _layout = setup_vgroupbox(text="Interaction Tools:")

        self.interaction_button = setup_vswitch(
            _layout,
            options=["Point", "BBox", "Scribble", "Lasso"],
            function=self.on_interaction_selected,
            fixed_color="rgb(0,100, 167)",
        )

        setup_icon(self.interaction_button.buttons[0], "new_points", theme=self._viewer.theme)
        setup_icon(self.interaction_button.buttons[1], "rectangle", theme=self._viewer.theme)
        setup_icon(self.interaction_button.buttons[2], "paint", theme=self._viewer.theme)
        setup_icon(self.interaction_button.buttons[3], "polygon_lasso", theme=self._viewer.theme)

        for i, shortcut in enumerate(["P", "B", "S", "L"]):
            button = self.interaction_button.buttons[i]
            key = QShortcut(QKeySequence(shortcut), button)
            key.activated.connect(lambda idx=i: self.on_interaction_shortcut(idx))
            button.setToolTip(f"press {shortcut} to toggle")
            # Show the shortcut on the label; the switch's options/value stay clean.
            button.setText(f"{button.text()} ({shortcut})")

        _group_box.setLayout(_layout)
        return _group_box

    def _init_inference_options(self) -> QGroupBox:
        """Inference-time options for the Settings tab (currently just Auto-zoom)."""
        _group_box, _layout = setup_vgroupbox(text="Inference:")

        self.propagate_ckbx = setup_checkbox(
            _layout,
            "Auto-zoom",
            True,
            function=self.on_propagate_ckbx,
            tooltips="Automatically zoom in and refine after each interaction. "
            "Disabled automatically when running on CPU.",
        )

        _group_box.setLayout(_layout)
        return _group_box

    def _init_label_aggregation(self) -> QGroupBox:
        """Label-aggregation controls for the Interact tab.

        Decide how a finished object is stored when you move to the next one (read at
        commit time in ``LayerControls._store_current_object``, so all of these can be
        changed mid-session):

        * **Output** — a separate layer per object, one shared **instance map** (each
          object gets its own label id), or one shared **semantic map** (every object is
          written with the fixed Class ID, so several instances share a semantic class).
        * **On overlap** (map modes only) — keep earlier objects, or let the new one
          overwrite them where they overlap.
        * **Class ID** (semantic mode only) — the label value written to the semantic map;
          change it between classes to assign instances to different semantic classes.
        """
        _group_box, _layout = setup_vgroupbox(text="Label Aggregation:")

        # --- Output mode --- #
        _out_layout = QHBoxLayout()
        _layout.addLayout(_out_layout)
        _out_label = setup_label(_out_layout, "Output:")
        _out_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.aggregation_output_combo = setup_combobox(
            _out_layout,
            options=[self.OUT_SEPARATE, self.OUT_INSTANCE, self.OUT_SEMANTIC],
        )
        _out_tips = [
            "Store each object as its own binary label layer (one layer per object).",
            "Merge all objects into a single instance map, each object written as its own "
            "label id.",
            "Merge all objects into a single semantic map, each object written with the "
            "fixed Class ID below, so several instances share one semantic class.",
        ]
        for _i, _tip in enumerate(_out_tips):
            self.aggregation_output_combo.setItemData(_i, _tip, Qt.ToolTipRole)

        # --- Overlap rule (map modes only) --- #
        _ovl_layout = QHBoxLayout()
        _layout.addLayout(_ovl_layout)
        self.aggregation_overlap_label = setup_label(_ovl_layout, "On overlap:")
        self.aggregation_overlap_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.aggregation_overlap_combo = setup_combobox(
            _ovl_layout,
            options=[self.OVERLAP_KEEP, self.OVERLAP_OVERWRITE],
            tooltips="keep existing: a new object only fills still-empty voxels, preserving "
            "earlier objects. overwrite: the new object replaces existing labels where they overlap.",
        )

        # --- Class ID (semantic mode only) --- #
        _id_layout = QHBoxLayout()
        _layout.addLayout(_id_layout)
        self.aggregation_class_id_label = setup_label(_id_layout, "Class ID:")
        self.aggregation_class_id_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.aggregation_class_id = setup_spinbox(_id_layout, maximum=999, default=1, stretch=1)
        self.aggregation_class_id.setToolTip(
            "Label value written to the semantic map; change it between classes to assign "
            "instances to different semantic classes."
        )
        self.aggregation_class_id.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Minimum)

        # Aggregation state is intentionally NOT persisted: every session starts at the
        # defaults (Separate layers / keep existing / Class ID 1). Set them explicitly -
        # before wiring the handlers - so a future reordering of the options can't change the
        # default, and so these initial sets don't fire the commit/enable logic.
        self.aggregation_output_combo.setCurrentText(self.OUT_SEPARATE)
        self.aggregation_overlap_combo.setCurrentText(self.OVERLAP_KEEP)
        self.aggregation_class_id.setValue(1)

        # Snapshot for the mid-object commit logic (see on_aggregation_changed).
        self._prev_agg_settings = self._read_agg_settings()

        self.aggregation_output_combo.currentTextChanged.connect(self.on_aggregation_output_changed)
        # Any aggregation change commits the in-flight object and starts a fresh segment.
        self.aggregation_output_combo.currentTextChanged.connect(self.on_aggregation_changed)
        self.aggregation_overlap_combo.currentTextChanged.connect(self.on_aggregation_changed)
        self.aggregation_class_id.valueChanged.connect(self.on_aggregation_changed)
        # Set the initial enabled state of the dependent controls for the default mode.
        self.on_aggregation_output_changed()

        _group_box.setLayout(_layout)
        return _group_box

    def _read_agg_settings(self) -> tuple:
        """Current (output, overlap, class_id) of the Label Aggregation controls."""
        return (
            self.aggregation_output_combo.currentText(),
            self.aggregation_overlap_combo.currentText(),
            int(self.aggregation_class_id.value()),
        )

    def _init_export_button(self) -> QGroupBox:
        """Initializes the export button"""
        _group_box, _layout = setup_vgroupbox(text="")

        self.export_button = setup_iconbutton(
            _layout, "Export", "pop_out", self._viewer.theme, self._export
        )
        _group_box.setLayout(_layout)
        return _group_box

    # Event Handlers
    def on_init_button(self, *args, **kwargs) -> None:
        """Toggle handler for the Initialize button: initialize when idle,
        uninitialize (tear down the live session) when one is already initialized."""
        if self._session_locked:
            self.on_uninit()
        else:
            self.on_init()

    def on_uninit(self, *args, **kwargs) -> None:
        """Placeholder for tearing down the live session (subclasses override)."""

    def on_init(self, *args, **kwargs) -> None:
        """Initializes the session configuration based on the selected model and image."""

    def on_image_selected(self):
        """When a new image is selected reset layers and session (cfg + gui)"""
        self._clear_layers()
        self._unlock_session()

    def on_model_selected(self):
        """When a new model is selected reset layers and session (cfg + gui)"""
        self._clear_layers()
        self._unlock_session()

    def on_mode_switched(self, *args, **kwargs) -> None:
        """Placeholder for switching between local and remote inference modes."""

    def on_connect_toggle(self, *args, **kwargs) -> None:
        """Placeholder for claiming or releasing a remote session."""

    def on_remote_settings_changed(self, *args, **kwargs) -> None:
        """Placeholder for handling changes to remote URL/API key fields."""

    def on_local_settings_changed(self, *args, **kwargs) -> None:
        """Placeholder for changes to baked-in local options (torch.compile / storage)."""

    def on_checkpoint_changed(self, *args, **kwargs) -> None:
        """Placeholder for edits to / clearing of the local checkpoint path."""

    def on_reset_interactions(self):
        """Reset only the current interaction"""
        self._clear_layers()

    def on_undo(self, *args, **kwargs) -> None:
        """Placeholder method for undoing the last interaction."""

    def on_next(self) -> None:
        """Resets the interactions."""
        print("_reset_interactions")

    def on_aggregation_output_changed(self, *args, **kwargs) -> None:
        """Enable On-overlap only for the map modes and Class ID only for the semantic mode."""
        output = self.aggregation_output_combo.currentText()
        is_map = output in (self.OUT_INSTANCE, self.OUT_SEMANTIC)
        is_semantic = output == self.OUT_SEMANTIC
        self.aggregation_overlap_combo.setEnabled(is_map)
        self.aggregation_overlap_label.setEnabled(is_map)
        self.aggregation_class_id.setEnabled(is_semantic)
        self.aggregation_class_id_label.setEnabled(is_semantic)

    def on_aggregation_changed(self, *args, **kwargs) -> None:
        """Placeholder: LayerControls commits the in-flight object when a setting changes."""
        self._prev_agg_settings = self._read_agg_settings()

    def on_prompt_selected(self, *args, **kwargs) -> None:
        """Placeholder method for when a prompt type is selected"""
        print("on_prompt_selected", self.prompt_button.index, self.prompt_button.value)

    def on_interaction_selected(self, *args, **kwargs) -> None:
        """Placeholder method for when an interaction type is selected."""
        print(
            "on_interaction_selected", self.interaction_button.index, self.interaction_button.value
        )

    def on_interaction_shortcut(self, idx: int) -> None:
        """Handle a P/B/S/L tool shortcut, toggling the tool on press.

        Pressing the shortcut of the tool that is already active deselects it so no
        further prompts are added; pressing any other shortcut activates its tool.
        """
        if self.interaction_button.index == idx:
            self.interaction_button._uncheck()
            self.on_interaction_deselected()
        else:
            self.interaction_button._on_button_pressed(idx)

    def on_interaction_deselected(self, *args, **kwargs) -> None:
        """Placeholder method for when the active interaction tool is deselected."""
        print("on_interaction_deselected")

    def on_propagate_ckbx(self, *args, **kwargs):
        print("on_propagate_ckbx", *args, **kwargs)

    def on_load_mask(self):
        pass

    def _export(self) -> None:
        """Placeholder method for exporting all generated label layers"""
