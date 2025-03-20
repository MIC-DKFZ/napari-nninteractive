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
    setup_spinbox,
    setup_vswitch,
)
from napari_toolkit.widgets.buttons.icon_button import setup_icon
from qtpy.QtCore import Qt
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QShortcut,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


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

        self._unlock_session()
        self._viewer.bind_key("Ctrl+Q", self._close, overwrite=True)

    # Base Behaviour
    def _close(self):
        """Closes the viewer and quits the application."""
        self._viewer.close()
        quit()

    def _unlock_session(self):
        """Unlocks the session, enabling model and image selection, and initializing controls."""
        self.init_button.setEnabled(True)

        self.reset_button.setEnabled(False)
        self.prompt_button.setEnabled(False)
        self.interaction_button.setEnabled(False)
        self.run_button.setEnabled(False)
        self.run_ckbx.setEnabled(False)
        self.export_button.setEnabled(False)
        self.reset_interaction_button.setEnabled(False)
        self.propagate_ckbx.setEnabled(False)
        self.label_for_init.setEnabled(False)
        self.class_for_init.setEnabled(False)
        self.auto_refine.setEnabled(False)
        # self.empty_mask_btn.setEnabled(False)
        self.load_mask_btn.setEnabled(False)
        self.add_button.setEnabled(False)
        self.add_ckbx.setEnabled(False)

    def _lock_session(self):
        """Locks the session, disabling model and image selection, and enabling control buttons."""
        self.init_button.setEnabled(False)

        self.reset_button.setEnabled(True)
        self.prompt_button.setEnabled(True)
        self.interaction_button.setEnabled(True)
        self.run_button.setEnabled(True)
        self.run_ckbx.setEnabled(True)
        self.export_button.setEnabled(True)
        self.reset_interaction_button.setEnabled(True)
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

        model_options = ["nnInteractive_v1.0"]

        self.model_selection = setup_combobox(
            _layout, options=model_options, function=self.on_model_selected
        )
        self.model_selection.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)

        _boxlayout = QHBoxLayout()
        _layout.addLayout(_boxlayout)
        self.model_selection_local = setup_lineedit(
            _boxlayout, placeholder="Use Local Checkpoint...", function=self.on_model_selected
        )

        def _reset_local_ckpt_lineedit():
            self.model_selection_local.setText("")
            self.on_model_selected()

        btn = setup_iconbutton(
            _boxlayout, "", "delete_shape", self._viewer.theme, function=_reset_local_ckpt_lineedit
        )
        btn.setFixedWidth(30)

        _group_box.setLayout(_layout)
        return _group_box

    def _init_image_selection(self) -> QGroupBox:
        """Initializes the image selection combo box in a group box."""
        _group_box, _layout = setup_vgroupbox(text="Image Selection:")

        self.image_selection = setup_layerselect(
            _layout, viewer=self._viewer, layer_type=Image, function=self.on_image_selected
        )
        self.image_selection.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)

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

        _group_box.setLayout(_layout)
        return _group_box

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
        self.class_for_init = setup_spinbox(h_layout, default=1, stretch=1)
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

    def on_reset_interactions(self):
        """Reset only the current interaction"""
        self._clear_layers()

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
