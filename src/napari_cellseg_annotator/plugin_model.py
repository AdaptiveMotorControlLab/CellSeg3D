import napari
from qtpy.QtWidgets import (
    QWidget,
)


class Model_Loader(QWidget):
    def __init__(self, viewer: "napari.viewer.Viewer", parent=None):

        super().__init__(parent)

        # self.master = parent
        self._viewer = viewer
        """napari.viewer.Viewer: viewer in which the widget is displayed"""

        self.data_path = ""
        self.results_path = ""

        # interface

        def build():
            return

        def load_dataset():
            return

        # loading images with MONAI
        # LoadImageD(keys="image", reader=WSIReader, backend="tifffile", level=1)
