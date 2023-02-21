from functools import partial

from qtpy.QtCore import QtWarningMsg
from qtpy.QtGui import QCursor
from qtpy.QtWidgets import QMenu


from napari_cellseg3d import interface as ui
from napari_cellseg3d.utils import Singleton

##################
# Screen size adjustment error handler
##################


def handle_adjust_errors(widget, type, context, msg: str):
    """Qt message handler that attempts to react to errors when setting the window size
    and resizes the main window"""
    head = msg.split(": ")[0]
    if type == QtWarningMsg and head == "QWindowsWindow::setGeometry":
        print(
            f"Qt resize error : {msg}\nhas been handled by attempting to resize the window"
        )
        try:
            if widget.parent() is not None:
                state = int(widget.parent().parent().windowState())
                if state == 0:  # normal state
                    widget.parent().parent().adjustSize()
                elif state == 2:  # maximized state
                    widget.parent().parent().showNormal()
                    widget.parent().parent().showMaximized()
        except RuntimeError:
            pass


def handle_adjust_errors_wrapper(widget):
    """Returns a callable that can be used with qInstallMessageHandler directly"""
    return partial(handle_adjust_errors, widget)


##################
# Context menu for utilities
##################


class UtilsDropdown(metaclass=Singleton):
    """Singleton class for use in instantiating only one Utility dropdown menu that can be accessed from the plugin."""

    caller_widget = None
    # TODO(cyril) : might cause issues with forcing all widget instances to remain forever

    def dropdown_menu_call(self, widget, event):
        """Calls the utility dropdown menu at the location of a CTRL+right-click"""
        # ### DEBUG ### #
        # print(event.modifiers)
        # print("menu call")
        # print(widget)
        # print(self)
        ##################
        if self.caller_widget is None:
            self.caller_widget = widget

        if event.button == 2 and "control" in event.modifiers:
            dragged = False
            yield
            # on move
            while event.type == "mouse_move":
                print(event.position)
                dragged = True
                yield
            # on release
            if dragged:
                # print("drag end")
                pass
            else:
                # print("clicked!")
                if widget is self.caller_widget:
                    # print(f"authorized widget {widget} to show menu")
                    pos = QCursor.pos()
                    self.show_utils_menu(widget, pos)
                # else:
                # print(f"blocked widget {widget} from opening utils")

    def show_utils_menu(self, widget, event):
        from napari_cellseg3d.plugin_utilities import UTILITIES_WIDGETS

        # print(self.parent().parent())
        # TODO create mapping for name:widget
        # menu = QMenu(self.parent().parent())
        menu = QMenu(widget.window())
        menu.setStyleSheet(
            f"background-color: {ui.napari_grey}; color: white;"
        )

        actions = []
        for title in UTILITIES_WIDGETS.keys():
            a = menu.addAction(f"Utilities : {title}")
            actions.append(a)

        action = menu.exec_(event)

        for possible_action in actions:
            if action == possible_action:
                text = possible_action.text().split(": ")[1]
                widget = UTILITIES_WIDGETS[text](widget._viewer)
                widget._viewer.window.add_dock_widget(widget)
