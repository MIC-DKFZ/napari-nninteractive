from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from .widget_main import nnInteractiveWidget

try:
    __version__ = _version("napari-nninteractive")
except PackageNotFoundError:
    __version__ = "unknown"


__all__ = ("nnInteractiveWidget",)
