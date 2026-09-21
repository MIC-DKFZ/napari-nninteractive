import napari
from napari._qt.layer_controls.qt_points_controls import QtPointsControls
from packaging.version import Version

from napari_nninteractive.controls import hide_widgets


class CustomQtPointsControls(QtPointsControls):
    def __init__(self, layer):
        super().__init__(layer)

        if Version(napari.__version__) >= Version("0.6.5"):
            hide_widgets(
                self,
                "_face_color_control.face_color_edit",
                "_face_color_control.face_color_label",
                "_border_color_control.border_color_edit",
                "_border_color_control.border_color_edit_label",
                "_symbol_combobox_control.symbol_combobox",
                "_symbol_combobox_control.symbol_combobox_label",
                "_text_visibility_control.text_disp_checkbox",
                "_text_visibility_control.text_disp_label",
                # Gone in napari 0.9, where projection mode replaced it.
                "_out_slice_checkbox_control.out_of_slice_checkbox",
                "_out_slice_checkbox_control.out_of_slice_checkbox_label",
            )
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
