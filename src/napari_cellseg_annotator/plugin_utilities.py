import napari
# Qt
from qtpy.QtWidgets import QTabWidget

from napari_cellseg_annotator.plugin_convert import ConvertUtils
# local
from napari_cellseg_annotator.plugin_crop import Cropping
from napari_cellseg_annotator.plugin_metrics import MetricsUtils


class Utilities(QTabWidget):
    def __init__(self, viewer: "napari.viewer.Viewer"):

        super().__init__(viewer)

        self._viewer = viewer

        self.cropping_tab = Cropping(viewer)
        self.metrics_tab = MetricsUtils(viewer)
        self.convert_tab = ConvertUtils(viewer)

        self.build()

    def build(self):

        self.addTab(self.cropping_tab, "Crop")
        self.addTab(self.metrics_tab, "Metrics")
        self.addTab(self.convert_tab, "Convert")
