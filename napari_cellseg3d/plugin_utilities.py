import napari

# Qt
from qtpy.QtWidgets import QWidget
from qtpy.QtWidgets import QVBoxLayout
from qtpy.QtWidgets import QSizePolicy

import napari_cellseg3d.interface as ui

# local
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

        names = ["crop", "aniso", "small", "inst", "sem", "thresh"]

        for key, name in zip(UTILITIES_WIDGETS, names):
            setattr(self, name, UTILITIES_WIDGETS[key](self._viewer))

        self.utils_widgets = []
        for n in names:
            wid = getattr(self, n)
            self.utils_widgets.append(wid)

        # self.crop = Cropping(self._viewer)
        # self.sem = ToSemanticUtils(self._viewer)
        # self.aniso = AnisoUtils(self._viewer)
        # self.inst = ToInstanceUtils(self._viewer)
        # self.thresh = ThresholdUtils(self._viewer)
        # self.small = RemoveSmallUtils(self._viewer)

        self.docked_widgets = []

        if len(self.utils_widgets) != len(UTILITIES_WIDGETS.keys()):
            raise RuntimeError(
                "One or several utility widgets are missing/erroneous"
            )
        # TODO how to auto-update list based on UTILITIES_WIDGETS ?

        self.utils_choice = ui.DropdownMenu(
            UTILITIES_WIDGETS.keys(), label="Utilities"
        )

        self._build()
        self.utils_choice.currentTextChanged.connect(self._update_visibility)
        self._dock_util()
        self._update_visibility()

    def _build(self):
        container = ui.ContainerWidget(parent=self)
        layout = container.layout
        layout.addWidget(self.utils_choice.label)
        layout.addWidget(self.utils_choice)

        # ui.add_widgets(layout, self.utils_widgets)
        self.setLayout(layout)
        container.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        self._update_visibility()

    def _update_visibility(self):
        widget_class = UTILITIES_WIDGETS[self.utils_choice.currentText()]

        for i in range(len(self.docked_widgets)):
            self.docked_widgets[i].setVisible(False)
            if isinstance(self.utils_widgets[i], widget_class):
                self.docked_widgets[i].setVisible(True)

    def _dock_util(self):
        for i in range(len(self.utils_widgets)):
            docked = self._viewer.window.add_dock_widget(
                widget=self.utils_widgets[i]
            )
            self.docked_widgets.append(docked)

    def remove_from_viewer(self):
        self._viewer.window.remove_dock_widget(self)
