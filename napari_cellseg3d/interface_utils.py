from functools import partial

from qtpy.QtCore import QtWarningMsg
from qtpy.QtGui import QCursor
from qtpy.QtWidgets import QMenu

from napari_cellseg3d import interface as ui

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


def context_menu_call(widget, event):
    # print(event.modifiers)
    print("RIGHT CLICK CALL")

    if event.button == 2 and "control" in event.modifiers:
        # print("mouse down")
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
            pos = QCursor.pos()
            show_utils_menu(widget, pos)


def show_utils_menu(widget, event):
    # from napari_cellseg3d.plugin_crop import Cropping
    # from napari_cellseg3d.plugin_convert import AnisoUtils
    # from napari_cellseg3d.plugin_convert import RemoveSmallUtils
    # from napari_cellseg3d.plugin_convert import ToInstanceUtils
    # from napari_cellseg3d.plugin_convert import ToSemanticUtils
    # from napari_cellseg3d.plugin_convert import ThresholdUtils
    from napari_cellseg3d.plugin_utilities import UTILITIES_WIDGETS

    # print(self.parent().parent())
    # TODO create mapping for name:widget
    # menu = QMenu(self.parent().parent())
    menu = QMenu(widget.window())
    menu.setStyleSheet(f"background-color: {ui.napari_grey}; color: white;")

    actions = []
    for title in UTILITIES_WIDGETS.keys():
        a = menu.addAction(f"Utilities : {title}")
        actions.append(a)

    # crop_action = menu.addAction("Utilities : Crop")
    # aniso_action = menu.addAction("Utilities: Correct anisotropy")
    # clear_small_action = menu.addAction("Utilities : Remove small objects")
    # to_instance_action = menu.addAction(
    #     "Utilities : Convert to instance labels"
    # )
    # to_semantic_action = menu.addAction(
    #     "Utilities : Convert to semantic labels"
    # )
    # thresh_action = menu.addAction("Utilities : Binarize")

    action = menu.exec_(event)

    for possible_action in actions:
        if action == possible_action:
            text = possible_action.text().split(": ")[1]
            widget = UTILITIES_WIDGETS[text](widget._viewer)
            widget._viewer.window.add_dock_widget(widget)

    # if action == crop_action:
    #     self._viewer.window.add_dock_widget(Cropping(self._viewer))
    #
    # if action == aniso_action:
    #     self._viewer.window.add_dock_widget(AnisoUtils(self._viewer))
    #
    # if action == clear_small_action:
    #     self._viewer.window.add_dock_widget(RemoveSmallUtils(self._viewer))
    #
    # if action == to_instance_action:
    #     self._viewer.window.add_dock_widget(ToInstanceUtils(self._viewer))
    #
    # if action == to_semantic_action:
    #     self._viewer.window.add_dock_widget(ToSemanticUtils(self._viewer))
    #
    # if action == thresh_action:
    #     self._viewer.window.add_dock_widget(ThresholdUtils(self._viewer))
