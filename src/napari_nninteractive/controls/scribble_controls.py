from napari._qt.layer_controls.qt_labels_controls import QtLabelsControls
from napari.utils.action_manager import action_manager
import napari
from packaging.version import Version


class CustomQtScribbleControls(QtLabelsControls):
    """Custom Qt controls for scribble layer, hiding not needed controls.

    Args:
        layer (Shapes): The shapes layer associated with this control panel.
    """

    def __init__(self, layer):
        super().__init__(layer)

        if Version(napari.__version__) >= Version("0.6.5"):
            fields_to_hide = [
                self._colormode_combobox_control.color_mode_combobox,
                self._colormode_combobox_control.color_mode_combobox_label,
                self._contour_spinbox_control.contour_spinbox,
                self._contour_spinbox_control.contour_spinbox_label,
                self._preserve_labels_checkbox_control.preserve_labels_checkbox,
                self._preserve_labels_checkbox_control.preserve_labels_checkbox_label,
                self._contour_spinbox_control.contour_spinbox,
                self._contour_spinbox_control.contour_spinbox_label,
                self._ndim_spinbox_control.ndim_spinbox,
                self._ndim_spinbox_control.ndim_spinbox_label,
                self._contiguous_checkbox_control.contiguous_checkbox,
                self._contiguous_checkbox_control.contiguous_checkbox_label,
                self._display_selected_label_checkbox_control.selected_color_checkbox,
                self._display_selected_label_checkbox_control.selected_color_checkbox_label,
            ]

            for field in fields_to_hide:
                field.hide()
                field.setDisabled(True)
            self._label_control.label_color.setDisabled(True)

            buttons_to_hide = [
                {"button": self.colormap_update, "shortcut": None},
                {"button": self.pick_button, "shortcut": "napari:activate_labels_picker_mode"},
                {"button": self.polygon_button, "shortcut": "napari:activate_labels_polygon_mode"},
                {"button": self.fill_button, "shortcut": "napari:activate_labels_fill_mode"},
                {"button": self.erase_button, "shortcut": "napari:activate_labels_erase_mode"},
                {
                    "button": self.transform_button,
                    "shortcut": "napari:activate_labels_transform_mode",
                },
            ]

            for button in buttons_to_hide:
                button["button"].setDisabled(True)
                button["button"].hide()
                if button["shortcut"] is not None:
                    action_manager.unbind_shortcut(button["shortcut"])
        else:

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
                {
                    "button": self.transform_button,
                    "shortcut": "napari:activate_labels_transform_mode",
                },
            ]

            for button in buttons_to_hide:
                button["button"].setDisabled(True)
                button["button"].hide()
                if button["shortcut"] is not None:
                    action_manager.unbind_shortcut(button["shortcut"])

        self.paint_button.setChecked(True)
