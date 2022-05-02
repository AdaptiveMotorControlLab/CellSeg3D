from napari_cellseg_annotator.plugin_base import BasePluginFolder


class MetricsUtils(BasePluginFolder):
    def __init__(self, viewer: "napari.viewer.Viewer"):

        super().__init__(viewer)

        self._viewer = viewer

    def build(self):

        self.btn_image_files.setText("Ground truth")
