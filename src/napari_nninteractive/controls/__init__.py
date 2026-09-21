"""Custom layer controls, which trim napari's stock panels down to what nnInteractive needs."""


def hide_widgets(owner, *paths: str) -> None:
    """Hide and disable the widgets of `owner` addressed by dotted attribute paths.

    napari keeps reorganizing the private widgets behind its layer controls, and
    occasionally drops one entirely (0.9 removed the points out-of-slice checkbox).
    Addressing them by name lets a path that no longer resolves be skipped instead
    of raising AttributeError while the control panel is being built - a leftover
    widget in the panel is a far better outcome than the layer failing to open.
    """
    for path in paths:
        widget = owner
        for attr in path.split("."):
            widget = getattr(widget, attr, None)
            if widget is None:
                break
        else:
            widget.hide()
            widget.setDisabled(True)
