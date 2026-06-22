from typing import Optional

from napari.layers import Image, Labels
from napari.viewer import Viewer
from napari_toolkit.containers import setup_vcollapsiblegroupbox, setup_vgroupbox, setup_vscrollarea
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
    QShortcut,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from napari_nninteractive._version_check import VersionChecker, _is_outdated


class BaseGUI(QWidget):
    """
    A base GUI class for building the Base GUI and connect the components with the correct functions.

    Args:
        viewer (Viewer): The Napari viewer instance to connect with the GUI.
        parent (Optional[QWidget], optional): The parent widget. Defaults to None.
    """

    def __init__(self, viewer: Viewer, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._width = 300
        self.setMinimumWidth(self._width)
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self._viewer = viewer
        self.session_cfg = None
        self._remote_mode = False
        self._settings = QSettings("MIC-DKFZ", "napari-nninteractive")

        _main_layout = QVBoxLayout()
        self.setLayout(_main_layout)

        _scroll_widget, _scroll_layout = setup_vscrollarea(_main_layout)

        _scroll_layout.addWidget(self._init_model_selection())  # Model Selection
        _scroll_layout.addWidget(self._init_image_selection())  # Image Selection
        _scroll_layout.addWidget(self._init_control_buttons())  # Init and Reset Button
        _scroll_layout.addWidget(self._init_init_buttons())  # Init and Reset Button
        _scroll_layout.addWidget(self._init_prompt_selection())  # Prompt Selection
        _scroll_layout.addWidget(self._init_interaction_selection())  # Interaction Selection
        _scroll_layout.addWidget(self._init_run_button())  # Run Button
        _scroll_layout.addWidget(self._init_export_button())  # Run Button

        _ = setup_acknowledgements(_scroll_layout, width=self._width)  # Acknowledgements

        # Update notice, below the logo (filled in asynchronously once PyPI has been queried).
        self.version_status_label = QLabel("")
        self.version_status_label.setWordWrap(True)
        self.version_status_label.setAlignment(Qt.AlignLeft)
        # Let the user select/copy the update command with the mouse or keyboard.
        self.version_status_label.setTextInteractionFlags(
            Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard
        )
        self.version_status_label.setVisible(False)
        _scroll_layout.addWidget(self.version_status_label)

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
            self.version_status_label.setText(
                "Update available. Please run:\n"
                "pip install -U nnInteractive napari-nninteractive"
            )
            self.version_status_label.setStyleSheet("color: #e8830c; font-weight: bold;")  # orange
        else:
            self.version_status_label.setText("nnInteractive is up to date")
            self.version_status_label.setStyleSheet("color: #2e9e2e;")  # green

    def _unlock_session(self):
        """Unlocks the session, enabling model and image selection, and initializing controls."""
        self.init_button.setEnabled(True)

        # Reset interaction capabilities until a checkpoint is loaded.
        self._set_interaction_button_support({0: True, 1: True, 2: True, 3: True})

        self.reset_button.setEnabled(False)
        self.instance_aggregation_ckbx.setEnabled(False)
        self.prompt_button.setEnabled(False)
        self.interaction_button.setEnabled(False)
        self.run_button.setEnabled(False)
        self.run_ckbx.setEnabled(False)
        self.export_button.setEnabled(False)
        self.reset_interaction_button.setEnabled(False)
        self.undo_button.setEnabled(False)
        self.propagate_ckbx.setEnabled(False)
        self.label_for_init.setEnabled(False)
        self.class_for_init.setEnabled(False)
        self.auto_refine.setEnabled(False)
        # self.empty_mask_btn.setEnabled(False)
        self.load_mask_btn.setEnabled(False)
        self.add_button.setEnabled(False)
        self.add_ckbx.setEnabled(False)

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
        self.init_button.setEnabled(False)

        self.reset_button.setEnabled(True)
        self.instance_aggregation_ckbx.setEnabled(True)
        self.prompt_button.setEnabled(True)
        self.interaction_button.setEnabled(True)
        self.run_button.setEnabled(True)
        self.run_ckbx.setEnabled(True)
        self.export_button.setEnabled(True)
        self.reset_interaction_button.setEnabled(True)
        self.undo_button.setEnabled(True)
        self.propagate_ckbx.setEnabled(True)
        self.label_for_init.setEnabled(True)
        self.class_for_init.setEnabled(True)
        self.auto_refine.setEnabled(True)
        # self.empty_mask_btn.setEnabled(True)
        self.load_mask_btn.setEnabled(True)
        self.add_button.setEnabled(True)
        self.add_ckbx.setEnabled(True)

    def _clear_layers(self):
        """Abstract function to clear all needed layers"""

    def _init_model_selection(self) -> QGroupBox:
        """Initializes the model selection as a combo box."""
        _group_box, _layout = setup_vgroupbox(text="Model Selection:")

        # Local | Remote mode switch
        self.mode_switch = setup_hswitch(
            _layout,
            options=["Local", "Remote"],
            function=self.on_mode_switched,
            default=0,
            fixed_color="rgb(0,100, 167)",
            tooltips="Run inference locally or on a remote nninteractive-server",
        )

        # --- Local container --- #
        self.local_container = QWidget()
        _local_layout = QVBoxLayout()
        _local_layout.setContentsMargins(0, 0, 0, 0)
        self.local_container.setLayout(_local_layout)
        _layout.addWidget(self.local_container)

        model_options = ["nnInteractive_v1.0"]

        self.model_selection = setup_combobox(
            _local_layout, options=model_options, function=self.on_model_selected
        )

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
        # These are niche settings, so they live in a collapsible section that is folded
        # by default. The fold state and the chosen values are persisted via QSettings.
        advanced_collapsed = self._settings.value("advanced_collapsed", True, type=bool)
        self.advanced_box, _advanced_layout = setup_vcollapsiblegroupbox(
            _local_layout, text="Advanced", collapsed=advanced_collapsed
        )

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
            tooltips="Claim a session on the nninteractive-server",
        )
        self.connect_btn.setFixedWidth(110)

        self.remote_status_label = QLabel("not connected")
        self.remote_status_label.setWordWrap(True)
        _remote_layout.addWidget(self.remote_status_label)

        # Default: Local visible, Remote hidden
        self.remote_container.setVisible(False)

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
        self.server_url_edit.textChanged.connect(
            lambda t: self._settings.setValue("server_url", t)
        )

        # Persist the advanced options (fold state + chosen values) between sessions.
        self.advanced_box.toggled.connect(
            lambda expanded: self._settings.setValue("advanced_collapsed", not expanded)
        )
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
            self.on_init,
            tooltips="Initialize the Model and Image Pair",
        )

        # License of the loaded model, shown directly below Initialize once a
        # session is ready (set in on_init for both local and remote modes).
        self.model_license_label = QLabel("")
        self.model_license_label.setWordWrap(True)
        _layout.addWidget(self.model_license_label)

        self.undo_button = setup_iconbutton(
            _layout,
            "Undo",
            "step_left",
            self._viewer.theme,
            self.on_undo,
            tooltips="Undo the last interaction for the current object - press Ctrl+Z",
            shortcut="Ctrl+Z",
        )
        self.reset_interaction_button = setup_iconbutton(
            _layout,
            "Reset Object",
            "delete",
            self._viewer.theme,
            self.on_reset_interactions,
            tooltips="Keep Model and Image Pair, just reset the interactions for the current object  - press R",
            shortcut="R",
        )
        self.reset_button = setup_iconbutton(
            _layout,
            "Next Object",
            "step_right",
            self._viewer.theme,
            self.on_next,
            tooltips="Keep current segmentation and go to the next object - press M",
            shortcut="M",
        )

        self.instance_aggregation_ckbx = setup_checkbox(
            _layout,
            "Instance Aggregation",
            False,
            tooltips="If checked: Add all objects to a single layer. In the case of overlap newer objects overwrite older objects.\n"
            "Otherwise: Create a separate layer for each object. ",
        )

        _group_box.setLayout(_layout)
        return _group_box

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
        _group_box, _layout = setup_vgroupbox(text="Prompt Type:")

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

        self.propagate_ckbx = setup_checkbox(
            _layout,
            "Auto-zoom",
            True,
            function=self.on_propagate_ckbx,
        )

        for i, shortcut in enumerate(["P", "B", "S", "L"]):
            key = QShortcut(QKeySequence(shortcut), self.interaction_button.buttons[i])
            key.activated.connect(lambda idx=i: self.interaction_button._on_button_pressed(idx))
            self.interaction_button.buttons[i].setToolTip(f"press {shortcut}")

        _group_box.setLayout(_layout)
        return _group_box

    def _init_run_button(self) -> QGroupBox:
        """Initializes the run button and auto-run checkbox"""
        _group_box, _layout = setup_vcollapsiblegroupbox(text="Manual Control:", collapsed=True)

        h_layout = QHBoxLayout()
        _layout.addLayout(h_layout)

        self.add_button = setup_iconbutton(
            h_layout,
            "Add Interaction",
            "add",
            self._viewer.theme,
            self.add_interaction,
            tooltips="add the current interaction",
        )
        self.run_button = setup_iconbutton(
            h_layout,
            "Run",
            "right_arrow",
            self._viewer.theme,
            self.on_run,
            tooltips="Run the predict step",
        )

        self.run_ckbx = setup_checkbox(
            _layout,
            "Auto Run Prediction",
            True,
            tooltips="Run automatically after each interaction",
        )

        self.add_ckbx = setup_checkbox(
            _layout,
            "Auto Add Interaction",
            True,
            tooltips="Add interaction automatically to session",
        )

        _group_box.setLayout(_layout)
        return _group_box

    def _init_export_button(self) -> QGroupBox:
        """Initializes the export button"""
        _group_box, _layout = setup_vgroupbox(text="")

        self.export_button = setup_iconbutton(
            _layout, "Export", "pop_out", self._viewer.theme, self._export
        )
        _group_box.setLayout(_layout)
        return _group_box

    # Event Handlers
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

    def on_prompt_selected(self, *args, **kwargs) -> None:
        """Placeholder method for when a prompt type is selected"""
        print("on_prompt_selected", self.prompt_button.index, self.prompt_button.value)

    def on_interaction_selected(self, *args, **kwargs) -> None:
        """Placeholder method for when an interaction type is selected."""
        print(
            "on_interaction_selected", self.interaction_button.index, self.interaction_button.value
        )

    def on_run(self, *args, **kwargs) -> None:
        """Placeholder method for run operation"""
        print("on_run")

    def on_propagate_ckbx(self, *args, **kwargs):
        print("on_propagate_ckbx", *args, **kwargs)

    def on_load_mask(self):
        pass

    def add_mask_init_layer(self):
        pass

    def _export(self) -> None:
        """Placeholder method for exporting all generated label layers"""
