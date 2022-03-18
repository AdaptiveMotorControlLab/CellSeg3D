from napari_cellseg_annotator.plugin_loader import Loader


def test_loader(make_napari_viewer) :
    viewer = make_napari_viewer()
    widget = Loader(viewer)
    widget.show_dialog_mod()