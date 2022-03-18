from napari_cellseg_annotator.plugin_loader import Loader
import os


def test_loader(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = Loader(viewer)
    widget.opath = os.path.join(os.path.dirname(__file__), "test_res")
