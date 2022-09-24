import pathlib

import napari
from qtpy.QtCore import QSize
from qtpy.QtGui import QIcon
from qtpy.QtGui import QPixmap

# Qt
from qtpy.QtWidgets import QVBoxLayout
from qtpy.QtWidgets import QWidget
from qtpy.QtWidgets import QMenu

# local
from napari_cellseg3d import interface as ui


class Helper(QWidget):
    # widget testing
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()

        self.help_url = (
            "https://adaptivemotorcontrollab.github.io/cellseg3d-docs/"
        )

        self.about_url = "https://wysscenter.ch/advances/3d-computer-vision-for-brain-analysis"
        self.repo_url = "https://github.com/AdaptiveMotorControlLab/CellSeg3d"
        self._viewer = viewer

        path = pathlib.Path(__file__).parent.resolve()
        url = str(path) + "/res/logo_alpha.png"
        image = QPixmap(url)

        self.logo_label = ui.Button(func=lambda: ui.open_url(self.repo_url))
        self.logo_label.setIcon(QIcon(image))
        self.logo_label.setMinimumSize(200, 200)
        self.logo_label.setIconSize(QSize(200, 200))
        self.logo_label.setStyleSheet(
            "QPushButton { background-color: transparent }"
        )
        self.logo_label.setToolTip("Open Github page")

        self.info_label = ui.make_label(
            f"You are using napari-cellseg3d v.{'0.0.1rc4'}\n\n"
            f"Plugin for cell segmentation developed\n"
            f"by the Mathis Lab of Adaptive Motor Control\n\n"
            f"Code by :\nCyril Achard\nMaxime Vidal\nJessy Lauer\nMackenzie Mathis\n"
            f"\nReleased under the MIT license",
            self,
        )

        self.btn1 = ui.Button("Help...", lambda: ui.open_url(self.help_url))
        self.btn1.setToolTip("Go to documentation")

        self.btn2 = ui.Button("About...", lambda: ui.open_url(self.about_url))

        self.btnc = ui.Button("Close", self.remove_from_viewer)

        self.build()

        from qtpy.QtGui import QCursor

        @viewer.mouse_drag_callbacks.append
        def context_menu_call(viewer, event):
            # print(event.modifiers)

            if event.button == 2 and "control" in event.modifiers:
                print("mouse down")
                dragged = False
                yield
                # on move
                while event.type == "mouse_move":
                    print(event.position)
                    dragged = True
                    yield
                # on release
                if dragged:
                    print("drag end")
                else:
                    print("clicked!")
                    pos = QCursor.pos()
                    self.show_menu(pos)

    def show_menu(self, event):
        from napari_cellseg3d.plugin_crop import Cropping
        from napari_cellseg3d.plugin_convert import AnisoUtils
        from napari_cellseg3d.plugin_convert import RemoveSmallUtils
        from napari_cellseg3d.plugin_convert import ToInstanceUtils
        from napari_cellseg3d.plugin_convert import ToSemanticUtils
        from napari_cellseg3d.plugin_convert import ThresholdUtils
        from napari_cellseg3d.plugin_utilities import UTILITIES_WIDGETS

        # print(self.parent().parent())
        # TODO create mapping for name:widget

        # menu = QMenu(self.parent().parent())
        menu = QMenu(self.window())
        menu.setStyleSheet(
            f"background-color: {ui.napari_grey}; color: white;"
        )

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
                widget = UTILITIES_WIDGETS[text](self._viewer)
                self._viewer.window.add_dock_widget(widget)

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

    def build(self):
        vbox = QVBoxLayout()

        widgets = [
            self.logo_label,
            self.info_label,
            self.btn1,
            self.btn2,
            self.btnc,
        ]
        ui.add_widgets(vbox, widgets)
        self.setLayout(vbox)
        # self.show()
        # self._viewer.window.add_dock_widget(self, name="Help/About...", area="right")

    def remove_from_viewer(self):
        self._viewer.window.remove_dock_widget(self)
