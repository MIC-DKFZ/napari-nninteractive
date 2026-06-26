import numpy as np
from napari.layers import Labels
from napari.layers.base._base_constants import ActionType

from napari_nninteractive.layers.abstract_layer import BaseLayerClass


class ScribbleLayer(BaseLayerClass, Labels):
    """
    A scribble layer class that extends `BaseLayerClass` and `Labels`, with prompt-based color
    adjustments and custom drawing interactions. This class handles color management, adding
    scribble data, and executing the drawing interactions.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.colormap = {
            None: None,
            1: self.colors[self.prompt_index],
            2: self.colors_set[0],
            3: self.colors_set[1],
        }

        self._is_free = False
        self._last_dim_not_displayed = None
        self._last_slice_id = None
        self.mouse_drag_callbacks.append(self.on_draw)

    def replace_color(self, _color) -> None:
        """
        Replaces the color of the scribble label for the current prompt index.

        Args:
            _color (List[float]): The new RGBA color to apply to the current prompt.
        """
        self.colormap = {
            None: None,
            1: self.colors[self.prompt_index],
            2: self.colors_set[0],
            3: self.colors_set[1],
        }
        self.refresh()

    def _add(self, data, *arg, **kwargs) -> None:
        """We dont need this function here"""

    def run(self) -> None:
        """
        Finalizes the current scribble interaction, updating the label index and marking the layer as free.
        """
        # Recolor the just-drawn stroke (value 1 -> committed prompt value) on the
        # painted slice only. The stroke never spans more than the last painted slice
        # (on_draw undoes any pending stroke before a new one, and _commit_staged_history
        # records that slice), so this avoids scanning/writing the whole D×H×W volume.
        if self._last_slice_id is not None:
            idx = [slice(None)] * 3
            idx[self._last_dim_not_displayed] = self._last_slice_id
            slice_view = self.data[tuple(idx)]  # integer index on one axis -> a view
            slice_view[slice_view == 1] = self.prompt_index + 2  # in-place, writes back to self.data
        else:
            # Defensive fallback: run is normally called only after a commit recorded a slice.
            self.data[self.data == 1] = self.prompt_index + 2
        self._is_free = True
        self.refresh()

    def remove_last(self) -> None:
        """
        Undoes the last action, reverting the most recent scribble interaction.
        """
        self.undo()

    def _commit_staged_history(self) -> None:
        """
        Commits the current staged history for the layer and marks the action as finished.
        """
        super()._commit_staged_history()
        dim = int(self._slice_input.not_displayed[0])
        self._last_dim_not_displayed = dim
        sid = int(np.rint(float(self._data_slice.point[dim])))
        sid = int(np.clip(sid, 0, self.data.shape[dim] - 1))
        self._last_slice_id = sid
        self._is_free = False
        self.events.finished(action=ActionType.ADDED, value="ADD")

    def on_draw(self, *args, **kwargs) -> None:
        """Handles the drawing interaction for the layer. Removes previous scribbles if the layer is occupied."""
        if not self._is_free:
            self.remove_last()
            self.colormap = {
                None: None,
                1: self.colors[self.prompt_index],
                2: self.colors_set[0],
                3: self.colors_set[1],
            }
            self._is_free = True
            self.refresh()
        self.colormap = {
            None: None,
            1: self.colors[self.prompt_index],
            2: self.colors_set[0],
            3: self.colors_set[1],
        }

    def get_last(self):
        """
        Retrieves a small 3D crop of the last scribble interaction together with its bounding box.

        Returns:
            tuple[np.ndarray, list[list[int]]] | None:
                crop_3d – uint8 array of shape (1, H, W) / (H, 1, W) / (H, W, 1) containing
                           only the painted voxels within the tight bounding box of the stroke.
                bbox     – [[z0,z1],[y0,y1],[x0,x1]] half-open voxel intervals.
            None if no slice has been recorded yet or the slice is empty.
        """
        if self._last_slice_id is None:
            return None

        dim = self._last_dim_not_displayed
        sid = self._last_slice_id

        idx = [slice(None)] * 3
        idx[dim] = sid
        slice_2d = self.data[tuple(idx)]  # 2D view, no copy

        ij = np.argwhere(slice_2d == 1)
        if len(ij) == 0:
            return None

        lo = ij.min(axis=0)
        hi = ij.max(axis=0) + 1

        crop_2d = (slice_2d[lo[0] : hi[0], lo[1] : hi[1]] == 1).astype(np.uint8)
        crop_3d = np.expand_dims(crop_2d, axis=dim)

        displayed_dims = [d for d in range(3) if d != dim]
        bbox = [[0, 0], [0, 0], [0, 0]]
        bbox[dim] = [sid, sid + 1]
        bbox[displayed_dims[0]] = [int(lo[0]), int(hi[0])]
        bbox[displayed_dims[1]] = [int(lo[1]), int(hi[1])]

        return (crop_3d, bbox)
