import os
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
from napari._qt.layer_controls.qt_layer_controls_container import layer_to_controls
from napari.layers import Labels
from napari.layers.base._base_constants import ActionType
from napari.utils.colormaps import DirectLabelColormap
from napari.utils.notifications import show_warning
from napari.utils.transforms import Affine
from napari.viewer import Viewer
from qtpy.QtWidgets import QFileDialog, QWidget

from napari_nninteractive.controls.bbox_controls import CustomQtBBoxControls
from napari_nninteractive.controls.lasso_controls import CustomQtLassoControls
from napari_nninteractive.controls.point_controls import CustomQtPointsControls
from napari_nninteractive.controls.scribble_controls import CustomQtScribbleControls
from napari_nninteractive.layers.bbox_layer import BBoxLayer
from napari_nninteractive.layers.lasso_layer import LassoLayer
from napari_nninteractive.layers.point_layer import SinglePointLayer
from napari_nninteractive.layers.scribble_layer import ScribbleLayer
from napari_nninteractive.utils.affine import is_orthogonal
from napari_nninteractive.utils.utils import ColorMapper, determine_layer_index
from napari_nninteractive.widget_gui import BaseGUI

layer_to_controls[SinglePointLayer] = CustomQtPointsControls
layer_to_controls[BBoxLayer] = CustomQtBBoxControls
layer_to_controls[ScribbleLayer] = CustomQtScribbleControls
layer_to_controls[LassoLayer] = CustomQtLassoControls


class LayerControls(BaseGUI):
    """
    A class for managing and interacting with different layers in the viewer,
    specifically designed for point, bounding box, and scribble layers.

    Args:
        viewer (Viewer): The Napari viewer instance to which layers will be added.
        parent (Optional[QWidget], optional): The parent widget. Defaults to None.
    """

    def __init__(self, viewer: Viewer, parent: Optional[QWidget] = None):
        super().__init__(viewer, parent)
        self.point_layer_name = "nnInteractive - Point Layer"
        self.bbox_layer_name = "nnInteractive - BBox Layer"
        self.scribble_layer_name = "nnInteractive - Scribble Layer"
        self.lasso_layer_name = "nnInteractive - Lasso Layer"
        self.layer_dict = {
            0: self.point_layer_name,
            1: self.bbox_layer_name,
            2: self.scribble_layer_name,
            3: self.lasso_layer_name,
        }

        self.label_layer_name = "nnInteractive - Label Layer"
        self.colormap = ColorMapper(49, seed=0.5, background_value=0)
        self._scribble_brush_size = 5
        self.object_index = 0
        # Accumulated bbox (union of the backend's per-change bboxes) of the in-progress
        # object, so it can be merged into the instance map by touching only the sub-volume
        # it occupies instead of scanning the whole array. ``None`` until a change is tracked;
        # ``_object_bbox_reliable`` goes False whenever a change cannot be localized (a backend
        # that returns no bbox, a no-op predict, or a whole-buffer seed), which forces a safe
        # global merge for that object.
        self._current_object_bbox = None
        self._object_bbox_reliable = True
        # Names of the interaction layers that committed an interaction, newest last. Used by
        # on_undo to remove the visual marker of the most recently undone interaction. The
        # backend supports single-level undo, so only the top entry is ever undoable; older
        # entries stay because their interactions are still applied. None means an interaction
        # without a layer marker (e.g. Initialize with Mask).
        self._interaction_history = []

        self._viewer.layers.selection.events.active.connect(self.on_layer_selected)

    # Layer Handling
    def _clear_layers(self) -> None:
        """Removes all layers in the viewer that are managed by this class."""
        layer_names = list(self.layer_dict.values())
        for layer_name in layer_names:
            if layer_name in self._viewer.layers:
                self._viewer.layers.remove(layer_name)
        # The interaction markers are gone, so nothing is left to undo for them.
        self._interaction_history = []

    def add_point_layer(self) -> None:
        """Adds a single point layer to the viewer."""
        point_layer = SinglePointLayer(
            name=self.point_layer_name,
            ndim=self.session_cfg["ndim"],
            affine=self.session_cfg["affine"],
            scale=self.session_cfg["scale"],
            translate=self.session_cfg["translate"],
            rotate=self.session_cfg["rotate"],
            shear=self.session_cfg["shear"],
            metadata=self.session_cfg["metadata"],
            opacity=0.7,
            size=2,
            prompt_index=self.prompt_button.index,
        )

        # point_layer.size = 0.2
        point_layer.events.finished.connect(self.on_interaction)
        self._viewer.add_layer(point_layer)

    def add_bbox_layer(self) -> None:
        """Adds a bounding box layer to the viewer."""
        bbox_layer = BBoxLayer(
            name=self.bbox_layer_name,
            ndim=self.session_cfg["ndim"],
            affine=self.session_cfg["affine"],
            scale=self.session_cfg["scale"],
            translate=self.session_cfg["translate"],
            rotate=self.session_cfg["rotate"],
            shear=self.session_cfg["shear"],
            metadata=self.session_cfg["metadata"],
            prompt_index=self.prompt_button.index,
            opacity=0.3,
        )
        bbox_layer.events.data.connect(self.on_interaction)
        self._viewer.add_layer(bbox_layer)

    def add_scribble_layer(self) -> None:
        """Adds a scribble layer to the viewer with an initial blank data array."""
        _data = np.zeros(self.session_cfg["shape"], dtype=np.uint8)
        scribble_layer = ScribbleLayer(
            data=_data,
            name=self.scribble_layer_name,
            affine=self.session_cfg["affine"],
            scale=self.session_cfg["scale"],
            translate=self.session_cfg["translate"],
            rotate=self.session_cfg["rotate"],
            shear=self.session_cfg["shear"],
            metadata=self.session_cfg["metadata"],
            prompt_index=self.prompt_button.index,
        )

        scribble_layer.brush_size = self._scribble_brush_size

        scribble_layer.events.finished.connect(self.on_interaction)
        self._viewer.add_layer(scribble_layer)

    def add_lasso_layer(self) -> None:
        """Adds a lasso layer to the viewer."""
        lasso_layer = LassoLayer(
            shape=self.session_cfg["shape"],
            name=self.lasso_layer_name,
            ndim=self.session_cfg["ndim"],
            affine=self.session_cfg["affine"],
            scale=self.session_cfg["scale"],
            translate=self.session_cfg["translate"],
            rotate=self.session_cfg["rotate"],
            shear=self.session_cfg["shear"],
            metadata=self.session_cfg["metadata"],
            prompt_index=self.prompt_button.index,
            opacity=0.3,
        )
        lasso_layer.events.data.connect(self.on_interaction)
        self._viewer.add_layer(lasso_layer)

    def add_label_layer(self, data, name) -> None:
        """
        Check if a layer with the layer_name already exists. If yes rename this by adding an index
        and afterward create the layer
        :return:
        :rtype:
        """

        label_layer = Labels(
            data,
            # self._data_result,
            name=name,
            opacity=0.3,
            affine=self.session_cfg["affine"],
            scale=self.session_cfg["scale"],
            translate=self.session_cfg["translate"],
            rotate=self.session_cfg["rotate"],
            shear=self.session_cfg["shear"],
            # colormap=self.colormap[index],
            metadata=self.session_cfg["metadata"],
        )
        label_layer._source = self.session_cfg["source"]

        self._viewer.add_layer(label_layer)

    def add_mask_init_layer(self) -> None:
        """
        Check if a layer with the layer_name already exists. If yes rename this by adding an index
        and afterward create the layer
        :return:
        :rtype:
        """

        _layer_res = Labels(
            np.zeros_like(self._data_result),
            name=self.mask_init_layer_name,
            opacity=0.3,
            affine=self.session_cfg["affine"],
            scale=self.session_cfg["scale"],
            translate=self.session_cfg["translate"],
            rotate=self.session_cfg["rotate"],
            shear=self.session_cfg["shear"],
            metadata=self.session_cfg["metadata"],
        )
        _layer_res._source = self.session_cfg["source"]

        self._viewer.add_layer(_layer_res)

    def init_with_mask(self):
        _layer_data = self._viewer.layers[self.label_for_init.currentText()].data

        assert (
            _layer_data.shape == self.session_cfg["shape"]
        )  # Labels and Image should have same shape

        self._data_result = (_layer_data == self.class_for_init.value()).astype(np.uint8)
        self.session.set_target_buffer(self._data_result)
        self._viewer.layers[self.label_layer_name].data = self._data_result

    # Event Handlers
    def on_init(self, *args, **kwargs) -> None:
        """
        Initializes the session by configuring the selected model and image and creating a label layer.

        Retrieves the selected model and image names from the GUI, extracts relevant data from the
        image layer, and creates a corresponding label layer in the viewer.
        """
        # --- MODEL HANDLING --- #
        # Get all model and image from the GUI
        image_name = self.image_selection.currentText()

        if image_name == "":
            raise ValueError("No Image Layer selected")

        if self._remote_mode:
            # Remote: server already loaded the checkpoint at startup.
            model_name = "remote"
            self.checkpoint_path = None
            print(f"Using remote model at: {self.server_url_edit.text().strip()}")
        else:
            model_name_local = self.model_selection_local.text()
            if model_name_local != "" and Path(model_name_local).exists():
                # Use Local Checkpoint
                model_name = Path(model_name_local).name
                self.checkpoint_path = model_name_local
            else:
                # Resolve the selected official model through the nnInteractive backend.
                # It downloads the model on first use, reuses it afterwards, and works
                # offline once a model has been downloaded.
                from nnInteractive.model_management import ensure_model_available

                idx = self.model_selection.currentIndex()
                if idx < 0 or idx >= len(self._model_ids):
                    raise ValueError(
                        "No model selected. Pick a model from the dropdown or set a local "
                        "checkpoint path."
                    )
                model_name = self._model_ids[idx]
                self.checkpoint_path = ensure_model_available(model_name)
            print(f"Using Model {model_name} at : {self.checkpoint_path}")

        # --- DATA HANDLING --- #
        # Get everything we need from the image layer
        image_layer = self._viewer.layers[image_name]
        self.source_cfg = {
            "name": image_name,
            "model": model_name,
            "ndim": image_layer.ndim,
            "shape": image_layer.data.shape,
            "affine": image_layer.affine,
            "scale": image_layer.scale,
            "translate": image_layer.translate,
            "rotate": image_layer.rotate,
            "shear": image_layer.shear,
            "source": image_layer.source,
            "metadata": image_layer.metadata,
        }

        self.session_cfg = self.source_cfg.copy()

        # 1. Non - Othogonal Affine
        if not (
            is_orthogonal(
                self.source_cfg["affine"],
                image_layer.ndim,
                self._viewer.dims.order,
                self._viewer.dims.ndisplay,
            )
        ):
            show_warning(
                "Your data is non-orthogonal. This is not supported by napari. "
                "To fix this the direction and shear is ignored during visualizing which changes the appearance (only visual) of your data."
            )
            # 1. Make affine orthogonal -> ignore rotate and shear
            self.session_cfg["affine"] = Affine(
                scale=self.source_cfg["affine"].scale, translate=self.source_cfg["affine"].translate
            )
            # 2. Apply to Image Layer
            image_layer.affine = self.session_cfg["affine"]
            self._viewer.reset_view()

        # 1. Non - Othogonal Transforms
        # dummy affine to check if transforms are non-orthogonal
        _transform_matrix = Affine(
            scale=self.source_cfg["scale"],
            translate=self.source_cfg["translate"],
            rotate=self.source_cfg["rotate"],
            shear=self.source_cfg["shear"],
        )

        if not is_orthogonal(
            _transform_matrix,
            image_layer.ndim,
            self._viewer.dims.order,
            self._viewer.dims.ndisplay,
        ):
            show_warning(
                "Your data is non-orthogonal. This is not supported by napari. "
                "To fix this the direction and shear is ignored during visualizing which changes the appearance (only visual) of your data."
            )

            # 1. Make transforms orthogonal
            self.session_cfg["rotate"] = np.eye(self.source_cfg["ndim"])
            self.session_cfg["shear"] = np.zeros(self.source_cfg["ndim"])

            # 2. Apply to Image Layer
            image_layer.rotate = self.session_cfg["rotate"]
            image_layer.shear = self.session_cfg["shear"]
            self._viewer.reset_view()

        # 2. Convert 2D Data to dummy 3D Data
        if self.source_cfg["ndim"] == 2:
            self.session_cfg["ndim"] = 3
            self.session_cfg["shape"] = np.insert(self.session_cfg["shape"], 0, 1)

            # 1. to Affine
            self.session_cfg["affine"] = self.session_cfg["affine"].expand_dims([0])

            # 2. to Transforms
            self.session_cfg["scale"] = np.insert(self.session_cfg["scale"], 0, 1)
            self.session_cfg["translate"] = np.insert(self.session_cfg["translate"], 0, 0)
            if len(self.session_cfg["shear"]) == 1:
                self.session_cfg["shear"] = np.append(self.session_cfg["shear"], 0)
            self.session_cfg["shear"] = np.insert(self.session_cfg["shear"], 0, 0)
            _rot = np.eye(self.session_cfg["ndim"])
            _rot[-2:, -2:] = self.session_cfg["rotate"]
            self.session_cfg["rotate"] = _rot

        # Compute the overall spacing when considering both, affine and scale transform
        self.session_cfg["spacing"] = np.array(self.session_cfg["scale"]) * np.array(
            self.session_cfg["affine"].scale
        )

        # Decide whether to resume the previous segmentation. We only resume when
        # the _resume_after_reconnect flag is armed -- set after a connection loss
        # (_handle_session_expired) -- and only when the surviving label layer
        # belongs to the *same* image layer object (identity, not merely the same
        # shape, which would be brittle). The shape check is a cheap guard against
        # a layer that no longer matches.
        resume = (
            getattr(self, "_resume_after_reconnect", False)
            and self.label_layer_name in self._viewer.layers
            and self._resume_image_layer is image_layer
            and tuple(self._viewer.layers[self.label_layer_name].data.shape)
            == tuple(self.session_cfg["shape"])
        )

        if resume:
            # Keep the existing label layer, its data, colormap and object_index
            # so the user carries on where they left off. The target buffer must
            # alias the layer data so the backend's writes remain visible.
            self._data_result = self._viewer.layers[self.label_layer_name].data
        else:
            # Create the target label array and layer
            self._data_result = np.zeros(self.session_cfg["shape"], dtype=np.uint8)
            # Continue numbering/colors from any objects already segmented for this
            # image in a previous session instead of restarting at 1.
            self.object_index = self._next_object_index()
            if self.label_layer_name in self._viewer.layers:
                self._viewer.layers.remove(self.label_layer_name)
            self.add_label_layer(self._data_result, self.label_layer_name)
            # Colour the working layer to match where the counter resumes, so the
            # in-progress object already shows its eventual colour.
            self._viewer.layers[self.label_layer_name].colormap = self.colormap[self.object_index]

        # Pin the resume to this image object so a later reconnect can verify it
        # is the same image before resuming. _resuming is read by the subclass'
        # on_init to seed the new session.
        self._resume_image_layer = image_layer
        self._resuming = resume

        # Fresh image/model pair: nothing from a previous object is undoable, and the
        # in-progress object's tracked extent starts empty.
        self._interaction_history = []
        self._reset_object_bbox()

        # Lock the Session
        self._lock_session()

    def on_reset_interactions(self):
        """Reset only the current interaction"""
        super().on_reset_interactions()
        # The object was emptied, so its tracked extent restarts too.
        self._reset_object_bbox()
        self.on_layer_selected()

    def _next_object_index(self) -> int:
        """Continue object numbering from work already present in the viewer for
        this image, so a new session does not restart at 1 and collide with the
        names/colors of objects from a previous session. Considers both the
        per-object ``object N - <name>`` layers and the values of the aggregated
        ``instance map - <name>`` layer. Returns 0 when nothing is present.
        """
        name = self.session_cfg["name"]
        # determine_layer_index returns max(N) + 1 over the "object N - <name>"
        # layers (or 0 when there are none); object_index is that max, one less.
        highest = (
            determine_layer_index(
                [layer.name for layer in self._viewer.layers], "object ", f" - {name}"
            )
            - 1
        )
        inst_name = f"instance map - {name}"
        if inst_name in self._viewer.layers:
            highest = max(highest, int(self._viewer.layers[inst_name].data.max()))
        return max(highest, 0)

    def _reset_object_bbox(self) -> None:
        """Start tracking a fresh in-progress object's extent (call when a new object begins)."""
        self._current_object_bbox = None
        self._object_bbox_reliable = True

    def _accumulate_object_bbox(self, bbox) -> None:
        """Fold a backend-reported changed-region bbox into the current object's extent.

        ``bbox`` is a half-open ``[[lb, ub], ...]`` per spatial axis, already clipped to the
        target buffer so it indexes the label layer directly, or ``None`` when the change could
        not be localized -- in which case the object falls back to a whole-volume merge.
        """
        if bbox is None:
            self._object_bbox_reliable = False
            return
        if self._current_object_bbox is None:
            self._current_object_bbox = [list(b) for b in bbox]
        else:
            self._current_object_bbox = [
                [min(cur[0], new[0]), max(cur[1], new[1])]
                for cur, new in zip(self._current_object_bbox, bbox)
            ]

    def _store_current_object(self) -> None:
        """Promote the in-progress label layer to a finished object.

        The Settings > Label Aggregation mode decides how (read here at commit time,
        so it can be changed between objects):

        * ``Separate layers``: copy it to its own ``object N - <name>`` binary layer.
        * ``Single map (keep existing)``: merge it into the shared ``instance map -
          <name>`` layer at the object's index, writing only where the map is still
          background so earlier objects are preserved.
        * ``Single map (overwrite)``: merge it in the same way, but overwrite existing
          labels wherever objects overlap.

        Advances object_index so subsequent objects keep consistent numbering and
        colours. Shared by ``on_next`` and by the reset / export paths that fold in
        the in-progress object before it would otherwise be lost.
        """
        label_layer = self._viewer.layers[self.label_layer_name]
        object_value = self.object_index + 1
        mode = self.label_aggregation_combo.currentText()

        if mode == self.AGG_BINARY:
            _name = f"object {object_value} - {self.session_cfg['name']}"
            self.add_label_layer(label_layer.data.copy(), _name)
            self._viewer.layers[_name].colormap = self.colormap[self.object_index]
        else:
            # aggregate (keep) / aggregate (overwrite): merge into one instance map,
            # each object written as its own integer value.
            _inst_name = f"instance map - {self.session_cfg['name']}"
            if _inst_name not in self._viewer.layers:
                self.add_label_layer(np.zeros_like(label_layer.data), _inst_name)
            inst_layer = self._viewer.layers[_inst_name]

            # Merge only the sub-volume this object occupies, when we could track its
            # extent (union of the backend's changed-region bboxes). Slicing yields views,
            # so writing through them updates the instance map in place. Falls back to a
            # whole-volume merge when the extent is unknown (older backend, whole-buffer
            # seed, etc.). Outside the object's bbox its mask is all-zero, so a local merge
            # produces exactly the same result as a global one.
            if self._object_bbox_reliable and self._current_object_bbox is not None:
                slc = tuple(slice(int(lb), int(ub)) for lb, ub in self._current_object_bbox)
            else:
                slc = tuple(slice(None) for _ in range(label_layer.data.ndim))

            sub_obj = label_layer.data[slc]
            sub_inst = inst_layer.data[slc]
            object_mask = sub_obj == 1
            if mode == self.AGG_KEEP:
                # Preserve earlier objects: only paint still-background voxels.
                object_mask = object_mask & (sub_inst == 0)
            sub_inst[object_mask] = object_value

            # Colour each object in the instance map with the same per-object colour
            # its working-layer preview used. The preview stores the object as value 1
            # and colours it via self.colormap[object_index]; here the object is value
            # object_index+1, so map value v back to self.colormap[v-1] to keep colours
            # consistent across the commit. Rebuilt each time so newly added values (and
            # any picked up on resume) are covered.
            color_dict = {None: (0, 0, 0, 0), 0: (0, 0, 0, 0)}
            for _v in range(1, object_value + 1):
                color_dict[_v] = self.colormap[_v - 1][1]
            inst_layer.colormap = DirectLabelColormap(color_dict=color_dict)
            inst_layer.refresh()

        self.object_index += 1

    def on_next(self) -> None:
        """
        Prepares the next label layer for interactions in the viewer.

        Retrieves the index of the last labeled object, renames the current label layer with
        this index, unbinds the original data by creating a deep copy, and clears all interaction
        layers. A new label layer with an updated colormap is then added to the viewer.
        """
        # Store the current object and recolour the working layer for the next one.
        self._store_current_object()
        self._viewer.layers[self.label_layer_name].colormap = self.colormap[self.object_index]
        # The next object starts with a fresh (empty) tracked extent.
        self._reset_object_bbox()

        self._clear_layers()
        self.prompt_button._uncheck()
        self.prompt_button._check(0)

    def on_prompt_selected(self) -> None:
        """
        Updates the prompt index for each layer in the viewer based on the selected prompt.

        Iterates through the layers specified in `layer_dict`, sets the prompt index for each
        corresponding layer using the current prompt button selection, and refreshes each layer to
        apply the updated prompt.
        """
        for layer_name in self.layer_dict.values():
            if layer_name in self._viewer.layers:
                self._viewer.layers[layer_name].set_prompt(self.prompt_button.index)
                self._viewer.layers[layer_name].refresh()

    def on_interaction_selected(self) -> None:
        """
        Activates or creates a layer based on the selected interaction type.

        If a layer of the specified `interaction_type` already exists, it is activated;
        otherwise, a new layer is created.
        """
        self.interaction_type = self.interaction_button.index
        layer_name = self.layer_dict.get(self.interaction_type)

        if layer_name is not None and layer_name in self._viewer.layers:  # Activate the Layer
            self._viewer.layers.selection.clear()
            self._viewer.layers.selection.add(self._viewer.layers[layer_name])
            self._viewer.layers.selection.active = self._viewer.layers[layer_name]

            self._viewer.layers.selection.events.active(value=self._viewer.layers[layer_name])

        elif self.interaction_type == 0:  # Add Point Layer
            self.add_point_layer()
        elif self.interaction_type == 1:  # Add BBox Layer
            self.add_bbox_layer()
        elif self.interaction_type == 2:  # Add Scrible Layer
            self.add_scribble_layer()
        elif self.interaction_type == 3:  # Add Lasso Layer
            self.add_lasso_layer()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            self._viewer.window._qt_viewer.setFocus()

    def on_interaction_deselected(self) -> None:
        """Deactivate the current interaction tool.

        Clears the active interaction layer selection so the viewer returns to a
        pan/zoom state and no new prompts are added until a tool is selected again.
        Clearing the selection emits ``selection.events.active`` which routes through
        ``on_layer_selected`` to uncheck the tool button; the explicit reset here also
        covers the case where no interaction layer was active.
        """
        self.interaction_type = None
        self.interaction_button._uncheck()
        self._viewer.layers.selection.clear()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            self._viewer.window._qt_viewer.setFocus()

    def on_run(self):
        if self.session is not None:
            self.session._predict()
            self._viewer.layers[self.label_layer_name].refresh()

    def on_interaction(self, event: Any):
        # Interactions are always added automatically now that the Manual Control
        # section (with its Auto Add / Auto Run toggles) has been removed.
        if (
            event.action == ActionType.ADDED
            and not self._viewer.layers[event.source.name].is_free()
        ):
            self._viewer.layers[event.source.name].refresh()

            self.add_interaction()

    def on_layer_selected(self, *args, **kwargs) -> None:
        """
        Updates the interaction button and sets the `interaction_type` based on
        the currently selected layer in the viewer.

        Args:
            *args: Additional arguments for the method.
            **kwargs: Additional keyword arguments for the method.
        """
        _layer = self._viewer.layers.selection.active

        if _layer is None:
            key = None
        else:
            key = next((k for k, v in self.layer_dict.items() if v == _layer.name), None)

        self.interaction_type = key
        self.interaction_button._uncheck()
        self.interaction_button._check(self.interaction_type)

    # Inference Behaviour
    def inference(self, data: Any, index: int) -> None:
        """
        Performs inference on the provided data.

        Args:
            data: The data obtained from the layer's run method.
            index (int): The index of the layer type, corresponding to the layer_dict key.
        """
        print(
            f"Inference for interaction {index} and prompt {self.prompt_button.index == 0} and valid data {data is not None} "
        )

    def _export(self) -> None:
        """Export all Label layers belonging to the current image & model pair as separate files
        using the napari plugins"""
        _img_layer = self._viewer.layers[self.source_cfg["name"]]

        _path = _img_layer.source.path
        if _path is not None:
            # Get the dtype from the input file
            _img_file = Path(_path).name
            _dtype = ".nii.gz" if str(_img_file).endswith(".nii.gz") else Path(_img_file).suffix
            _output_file = _img_file.replace(_dtype, "")
        else:
            # If nothing is defined we save as .nii.gz
            _dtype = ".nii.gz"
            _output_file = self.source_cfg["name"] + _dtype

        _dialog = QFileDialog(self)
        _dialog.setDirectory(os.getcwd())

        _output_dir = _dialog.getExistingDirectory(
            self,
            "Select an Output Directory",
            options=QFileDialog.DontUseNativeDialog | QFileDialog.ShowDirsOnly,
        )

        if _output_dir == "":
            return

        elif Path(_output_dir).is_dir():
            # Fold any uncommitted in-progress object in first. In aggregate mode it is
            # only merged into the instance map on Next Object, so without this an
            # export mid-object would silently drop the last object. Done only after a
            # valid directory is chosen, so cancelling the dialog never commits.
            if self.label_layer_name in self._viewer.layers and np.any(
                self._viewer.layers[self.label_layer_name].data
            ):
                self.on_next()

            _output_dir = Path(_output_dir).joinpath(f"{_output_file}_nnInteractive")
            Path(_output_dir).mkdir(exist_ok=True)

            for _layer in self._viewer.layers:
                if _layer.name.startswith("object ") and _layer.name.endswith(
                    f" - {self.source_cfg['name']}"
                ):
                    _index = int(
                        _layer.name.replace("object ", "").replace(
                            f" - {self.source_cfg['name']}", ""
                        )
                    )
                    _file_name = f"{_output_file}_{str(_index).zfill(4)}{_dtype}"
                elif f"instance map - {self.session_cfg['name']}" == _layer.name:
                    _file_name = f"{_output_file}_instance_map{_dtype}"

                else:
                    continue

                # _file_name = f"{_output_file}_{str(_index).zfill(4)}{_dtype}"
                _file = str(Path(_output_dir).joinpath(_file_name))

                # reverse the corrections for non-orthogonal data and convert dummy 3d back to 2d
                _data = _layer.data[0] if self.source_cfg["ndim"] == 2 else _layer.data
                _layer_temp = Labels(
                    _data,
                    name="_temp",
                    affine=self.source_cfg["affine"],
                    scale=self.source_cfg["scale"],
                    translate=self.source_cfg["translate"],
                    rotate=self.source_cfg["rotate"],
                    shear=self.source_cfg["shear"],
                    metadata=self.source_cfg["metadata"],
                )

                _layer_temp._source = self.source_cfg["source"]
                _layer_temp.save(_file)
                del _layer_temp
        else:
            raise ValueError("Output path has to be a directory, not a file")
