from napari._qt.layer_controls.qt_shapes_controls import QtShapesControls
from napari.utils.action_manager import action_manager
import napari
from packaging.version import Version


class CustomQtLassoControls(QtShapesControls):
    """Custom Qt controls for bounding box layer, hiding controls for non-rectangle shapes.

    Args:
        layer (Shapes): The shapes layer associated with this control panel.
    """

    def __init__(self, layer):
        super().__init__(layer)

        # We don't need this Fields -> Hide them
        if Version(napari.__version__) >= Version("0.6.5"):
            fields_to_hide = [
                self._face_color_control.face_color_edit,
                self._edge_color_control.edge_color_edit,
                self._face_color_control.face_color_label,
                self._edge_color_control.edge_color_label,
            ]

            for field in fields_to_hide:
                field.hide()
                field.setDisabled(True)
        else:
            fields_to_hide = [self.faceColorEdit, self.edgeColorEdit]
            for field in fields_to_hide:
                label_item = self.layout().labelForField(field)
                field.hide()
                label_item.hide()
                field.setDisabled(True)

        # We don't need all these button -> hide and disable tem + remove key binding
        buttons_to_hide = [
            {"button": self.ellipse_button, "shortcut": "napari:activate_add_ellipse_mode"},
            {"button": self.polygon_button, "shortcut": "napari:activate_add_polygon_mode"},
            {
                "button": self.rectangle_button,
                "shortcut": "napari:activate_add_rectangle_mode",
            },
            {"button": self.line_button, "shortcut": "napari:activate_add_line_mode"},
            {"button": self.path_button, "shortcut": "napari:activate_add_path_mode"},
            {"button": self.vertex_insert_button, "shortcut": "napari:activate_vertex_insert_mode"},
            {"button": self.vertex_remove_button, "shortcut": "napari:activate_vertex_remove_mode"},
            {"button": self.move_front_button, "shortcut": "napari:move_shapes_selection_to_front"},
            {"button": self.move_back_button, "shortcut": "napari:move_shapes_selection_to_back"},
            {"button": self.transform_button, "shortcut": "napari:activate_shapes_transform_mode"},
            {"button": self.direct_button, "shortcut": "napari:activate_direct_mode"},
            {"button": self.polyline_button, "shortcut": "napari:activate_add_polyline_mode"},
        ]

        for button in buttons_to_hide:
            button["button"].setDisabled(True)
            button["button"].hide()
            action_manager.unbind_shortcut(button["shortcut"])

        # Reorder the remaining buttons to not have a sparse layout
        self.button_grid.addWidget(self.delete_button, 0, 1)
        self.button_grid.addWidget(self.polygon_lasso_button, 0, 2)

        self.polygon_lasso_button.setChecked(True)
