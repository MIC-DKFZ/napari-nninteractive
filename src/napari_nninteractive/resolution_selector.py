from __future__ import annotations

from typing import Sequence

import numpy as np
from qtpy.QtWidgets import (
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QRadioButton,
    QVBoxLayout,
)

RECOMMENDED_MAX_AXIS = 2000


def exceeds_limit(shape: Sequence[int]) -> bool:
    return max(int(s) for s in shape) > RECOMMENDED_MAX_AXIS


def recommended_level(shapes: Sequence[Sequence[int]]) -> int:
    for level, shape in enumerate(shapes):
        if not exceeds_limit(shape):
            return level
    return len(shapes) - 1


class ResolutionLevelDialog(QDialog):

    def __init__(self, shapes, scale=None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select resolution level")
        self._shapes = [tuple(int(s) for s in shp) for shp in shapes]

        layout = QVBoxLayout(self)
        intro = QLabel(
            "This image has multiple resolution levels. Choose the level to "
            "segment.\n\nFiner levels are more detailed but use far more memory. "
            f"nnInteractive becomes unreliable beyond about {RECOMMENDED_MAX_AXIS} "
            "voxels per axis, so a level within that range is recommended."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self._group = QButtonGroup(self)
        rec = recommended_level(self._shapes)
        for level, shape in enumerate(self._shapes):
            voxels = int(np.prod(shape, dtype=np.int64))
            dims = " x ".join(str(s) for s in shape)
            text = f"Level {level}:   {dims} px   ({voxels / 1e6:.1f} MV)"
            if exceeds_limit(shape):
                text += "   \u26a0 exceeds recommended size"
            if level == rec:
                text += "   (recommended)"
            btn = QRadioButton(text)
            if scale is not None:
                scale_arr = np.atleast_1d(np.asarray(scale, dtype=float))
                full = np.asarray(self._shapes[0], dtype=float)
                this = np.asarray(shape, dtype=float)
                level_scale = scale_arr * (full / this)
                phys = " x ".join(f"{s * sc:.1f}" for s, sc in zip(shape, level_scale))
                btn.setToolTip(f"Physical extent: {phys} (in layer scale units)")
            btn.setChecked(level == rec)
            self._group.addButton(btn, level)
            layout.addWidget(btn)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_level(self) -> int:
        """Index of the chosen level (0 = highest resolution)."""
        return self._group.checkedId()
