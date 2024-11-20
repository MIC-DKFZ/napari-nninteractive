from napari._qt.qt_resources import QColoredSVGIcon
from qtpy.QtCore import QEvent, QObject, QSize
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import QWidget


class IconUpdater(QObject):
    """A class that updates a widget's icon color based on its enabled state and theme."""

    def __init__(self, widget: QWidget, icon: QIcon, theme: str, *args, **kwargs):
        super().__init__(widget, *args, **kwargs)
        self.widget = widget
        self.icon = icon
        self.theme = theme

    def eventFilter(self, obj: QWidget, event: QEvent) -> bool:
        """Filters events for the widget to detect state changes.

        Args:
            obj (QWidget): The object where the event occurred.
            event (QEvent): The event to filter.

        Returns:
            bool: True if the event was handled, False otherwise.
        """
        if obj == self.widget and event.type() == QEvent.EnabledChange:
            self.update_icon(obj)
            return super().eventFilter(obj, event)

        else:
            return super().eventFilter(obj, event)

    def update_icon(self, obj: QWidget) -> None:
        """Updates the icon color based on the widget's state."""
        palette = obj.palette()
        if self.widget.isEnabled():
            color = palette.color(palette.ButtonText)
        else:
            color = palette.color(palette.Dark)
        updated_icon = self.icon.colored(color=color.name())
        self.widget.setIcon(updated_icon)


def setup_icon(_widget: QWidget, icon_name: str, theme: str = "dark", size: int = 24) -> QIcon:
    """Sets up an icon for a widget, including dynamic color updates.

    Args:
        _widget (QWidget): The widget to set up the icon for.
        icon_name (str): The resource name of the icon.
        theme (str, optional): The theme to use for the icon ("dark" or "light"). Defaults to "dark".
        size (int, optional): The size of the icon in pixels. Defaults to 24.

    Returns:
        QIcon: The configured icon for the widget.
    """
    _icon = QColoredSVGIcon.from_resources(icon_name)
    _icon = _icon.colored(theme=theme)

    _widget.setIcon(_icon)
    _widget.setIconSize(QSize(size, size))

    _widget.installEventFilter(IconUpdater(_widget, _icon, theme=theme))

    return _icon
