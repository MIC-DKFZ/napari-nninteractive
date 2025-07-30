from napari._qt.layer_controls.qt_labels_controls import QtLabelsControls
from napari.utils.action_manager import action_manager


class CustomQtScribbleControls(QtLabelsControls):
    """Custom Qt controls for scribble layer, hiding not needed controls.

    Args:
        layer (Shapes): The shapes layer associated with this control panel.
    """

    def __init__(self, layer):
        super().__init__(layer)

        fields_to_hide = [
            self.colorModeComboBox,
            self.contigCheckBox,
            self.preserveLabelsCheckBox,
            self.selectedColorCheckbox,
            # self.blendComboBox,
            # self.brushSizeSlider,
            self.contourSpinBox,
            self.ndimSpinBox,
        ]

        for field in fields_to_hide:
            label_item = self.layout().labelForField(field)
            field.hide()
            if label_item is not None:
                label_item.hide()
                field.setDisabled(True)

        self.selectionSpinBox.setDisabled(True)

        buttons_to_hide = [
            {"button": self.colormapUpdate, "shortcut": None},
            {"button": self.pick_button, "shortcut": "napari:activate_labels_picker_mode"},
            {"button": self.polygon_button, "shortcut": "napari:activate_labels_polygon_mode"},
            {"button": self.fill_button, "shortcut": "napari:activate_labels_fill_mode"},
            {"button": self.erase_button, "shortcut": "napari:activate_labels_erase_mode"},
            {"button": self.transform_button, "shortcut": "napari:activate_labels_transform_mode"},
        ]

        for button in buttons_to_hide:
            button["button"].setDisabled(True)
            button["button"].hide()
            if button["shortcut"] is not None:
                action_manager.unbind_shortcut(button["shortcut"])

        self.paint_button.setChecked(True)
