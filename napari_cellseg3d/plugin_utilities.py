import napari

# Qt
from qtpy.QtCore import qInstallMessageHandler
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QSizePolicy
from qtpy.QtWidgets import QLayout
from qtpy.QtWidgets import QVBoxLayout
from qtpy.QtWidgets import QWidget

# local
import napari_cellseg3d.interface as ui
from napari_cellseg3d.interface_utils import handle_adjust_errors_wrapper
from napari_cellseg3d.plugin_crop import Cropping
from napari_cellseg3d.plugin_convert import AnisoUtils
from napari_cellseg3d.plugin_convert import RemoveSmallUtils
from napari_cellseg3d.plugin_convert import ToInstanceUtils
from napari_cellseg3d.plugin_convert import ToSemanticUtils
from napari_cellseg3d.plugin_convert import ThresholdUtils
from napari_cellseg3d.plugin_metrics import MetricsUtils

UTILITIES_WIDGETS = {
    "Crop": Cropping,
    "Correct anisotropy": AnisoUtils,
    "Remove small objects": RemoveSmallUtils,
    "Convert to instance labels": ToInstanceUtils,
    "Convert to semantic labels": ToSemanticUtils,
    "Threshold": ThresholdUtils,
}


class Utilities(QWidget):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer

        attr_names = ["crop", "aniso", "small", "inst", "sem", "thresh"]
        self._create_utils_widgets(attr_names)

        # self.crop = Cropping(self._viewer)
        # self.sem = ToSemanticUtils(self._viewer)
        # self.aniso = AnisoUtils(self._viewer)
        # self.inst = ToInstanceUtils(self._viewer)
        # self.thresh = ThresholdUtils(self._viewer)
        # self.small = RemoveSmallUtils(self._viewer)

        self.utils_choice = ui.DropdownMenu(
            UTILITIES_WIDGETS.keys(), label="Utilities"
        )

        self._build()
        self.utils_choice.currentIndexChanged.connect(self._update_visibility)
        # self._dock_util()
        self._update_visibility()
        qInstallMessageHandler(handle_adjust_errors_wrapper(self))

    def _build(self):

        layout = QVBoxLayout()
        ui.add_widgets(layout, self.utils_widgets)
        layout.addWidget(self.utils_choice.label, alignment=ui.BOTT_AL)
        layout.addWidget(self.utils_choice, alignment=ui.BOTT_AL)

        layout.setSizeConstraint(QLayout.SetFixedSize)
        self.setLayout(layout)
        self.setMinimumHeight(500)
        self.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
        self._update_visibility()

    def _create_utils_widgets(self, names):
        for key, name in zip(UTILITIES_WIDGETS, names):
            setattr(self, name, UTILITIES_WIDGETS[key](self._viewer))

        self.utils_widgets = []
        for n in names:
            wid = getattr(self, n)
            self.utils_widgets.append(wid)

        if len(self.utils_widgets) != len(UTILITIES_WIDGETS.keys()):
            raise RuntimeError(
                "One or several utility widgets are missing/erroneous"
            )
        # TODO how to auto-update list based on UTILITIES_WIDGETS ?

    def _update_visibility(self):
        widget_class = UTILITIES_WIDGETS[self.utils_choice.currentText()]
        # print("vis. updated")
        # print(self.utils_widgets)
        self._hide_all()
        for i, w in enumerate(self.utils_widgets):
            if isinstance(w, widget_class):
                w.setVisible(True)
                w.adjustSize()
            # else:
            #     self.utils_widgets[i].setVisible(False)

    def _hide_all(self):
        for w in self.utils_widgets:
            w.setVisible(False)
        # self.setWindowState(Qt.WindowMaximized)
        # if self.parent() is not None:
        # print(self.parent().windowState())
        # print(int(self.parent().parent().windowState()))
        # self.parent().parent().showNormal()
        # self.parent().parent().showMaximized()
        # state = int(self.parent().parent().windowState())
        # if state == 0:
        # self.parent().parent().adjustSize()
        # elif state == 2:
        # self.parent().parent().showNormal()
        # self.parent().parent().showMaximized()
        # pass

    # def _dock_util(self):
    #     for i in range(len(self.utils_widgets)):
    #         docked = self._viewer.window.add_dock_widget(
    #             widget=self.utils_widgets[i]
    #         )
    #         self.docked_widgets.append(docked)

    # def remove_from_viewer(self):
    #     self._viewer.window.remove_dock_widget(self)
