"""Non-blocking check for newer releases of the plugin and its backend on PyPI.

The network request runs in a background daemon thread so it never blocks GUI
startup, and the result is delivered back on the GUI thread via a Qt signal.
The check is best effort: when PyPI is unreachable (offline, behind a firewall,
etc.) it simply stays silent instead of nagging the user.
"""

import json
import threading
from importlib.metadata import PackageNotFoundError, version
from urllib.request import urlopen

from qtpy.QtCore import QObject, Signal

# PyPI project names to check, in display order.
PACKAGES = ("napari-nninteractive", "nnInteractive")

try:
    from packaging.version import InvalidVersion
    from packaging.version import parse as _parse_version
except ImportError:  # packaging is virtually always present, but degrade gracefully
    _parse_version = None
    InvalidVersion = Exception


def _installed_version(package: str):
    """Return the installed version string, or None if the package is missing."""
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _latest_version(package: str, timeout: float = 5.0) -> str:
    """Return the latest release version for `package` from the PyPI JSON API."""
    url = f"https://pypi.org/pypi/{package}/json"
    with urlopen(url, timeout=timeout) as response:  # noqa: S310 - fixed https URL
        data = json.load(response)
    return data["info"]["version"]


def _is_outdated(installed: str, latest: str) -> bool:
    """True if `installed` is an older release than `latest`."""
    if _parse_version is not None:
        try:
            return _parse_version(installed) < _parse_version(latest)
        except InvalidVersion:
            return False
    # Fallback when `packaging` is unavailable: only flag an exact mismatch.
    return installed != latest


class VersionChecker(QObject):
    """Checks PyPI for newer releases in a background daemon thread.

    Connect to `finished`, then call `start()`. The signal carries a dict mapping
    each package name to an `(installed, latest)` tuple; either entry may be None
    (package not installed, or PyPI could not be reached). It is emitted from the
    worker thread, so Qt delivers it to GUI-thread slots via a queued connection.
    """

    # Emits {package_name: (installed_or_None, latest_or_None)}
    finished = Signal(object)

    def start(self) -> None:
        threading.Thread(target=self._run, name="nni-version-check", daemon=True).start()

    def _run(self) -> None:
        results = {}
        for package in PACKAGES:
            installed = _installed_version(package)
            try:
                latest = _latest_version(package)
            except Exception:  # noqa: BLE001 - offline / PyPI down / bad payload: skip silently
                latest = None
            results[package] = (installed, latest)
        self.finished.emit(results)
