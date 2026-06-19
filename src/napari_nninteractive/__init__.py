from importlib.metadata import PackageNotFoundError, version

from .widget_main import nnInteractiveWidget

try:
    __version__ = version("napari-nninteractive")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = ("nnInteractiveWidget",)
