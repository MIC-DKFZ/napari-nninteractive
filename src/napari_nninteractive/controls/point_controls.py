from napari._qt.layer_controls.qt_points_controls import QtPointsControls
import napari
from packaging.version import Version


class CustomQtPointsControls(QtPointsControls):
    def __init__(self, layer):
        super().__init__(layer)

        if Version(napari.__version__) >= Version("0.6.5"):
            fields_to_hide = [
                self._face_color_control.face_color_edit,
                self._face_color_control.face_color_label,
                self._border_color_control.border_color_edit,
                self._border_color_control.border_color_edit_label,
                self._symbol_combobox_control.symbol_combobox,
                self._symbol_combobox_control.symbol_combobox_label,
                self._text_visibility_control.text_disp_checkbox,
                self._text_visibility_control.text_disp_label,
                self._out_slice_checkbox_control.out_of_slice_checkbox,
                self._out_slice_checkbox_control.out_of_slice_checkbox_label,
            ]

            for field in fields_to_hide:
                field.hide()
                field.setDisabled(True)
        else:
            fields_to_hide = [
                self.faceColorEdit,
                self.borderColorEdit,
                self.symbolComboBox,
                self.textDispCheckBox,
                self.outOfSliceCheckBox,
            ]
            for field in fields_to_hide:
                label_item = self.layout().labelForField(field)
                field.hide()
                label_item.hide()
                field.setDisabled(True)

        self.addition_button.setChecked(True)
