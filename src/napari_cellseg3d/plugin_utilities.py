import napari
# Qt
from qtpy.QtWidgets import QTabWidget

from napari_cellseg3d.plugin_convert import ConvertUtils
# local
from napari_cellseg3d.plugin_crop import Cropping
from napari_cellseg3d.plugin_metrics import MetricsUtils


class Utilities(QTabWidget):
    def __init__(self, viewer: "napari.viewer.Viewer"):

        super().__init__()

        self._viewer = viewer

        self.cropping_tab = Cropping(viewer, parent=self)
        self.metrics_tab = MetricsUtils(viewer, parent=self)
        self.convert_tab = ConvertUtils(viewer, parent=self)

        self.build()

    def build(self):

        self.addTab(self.convert_tab, "Convert")
        self.addTab(self.metrics_tab, "Metrics")
        self.addTab(self.cropping_tab, "Crop")

        self.setBaseSize(230, 150)
        self.setMinimumSize(230, 100)

    def remove_from_viewer(self):
        self._viewer.window.remove_dock_widget(self)
