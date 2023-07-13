from pathlib import Path

from napari_cellseg3d import plugins
from napari_cellseg3d.code_plugins import plugin_metrics as m


def test_all_plugins_import(make_napari_viewer_proxy):
    plugins.napari_experimental_provide_dock_widget()


def test_plugin_metrics(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    w = m.MetricsUtils(viewer=viewer, parent=None)
    viewer.window.add_dock_widget(w)

    im_path = str(Path(__file__).resolve().parent / "res/test.tif")
    labels_path = im_path

    w.image_filewidget.text_field = im_path
    w.labels_filewidget.text_field = labels_path
    w.compute_dice()
