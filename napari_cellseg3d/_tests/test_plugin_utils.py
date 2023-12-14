import numpy as np

from napari_cellseg3d.code_plugins.plugin_convert import StatsUtils
from napari_cellseg3d.code_plugins.plugin_crop import Cropping
from napari_cellseg3d.code_plugins.plugin_utilities import (
    UTILITIES_WIDGETS,
    Utilities,
)
from napari_cellseg3d.utils import rand_gen


def test_utils_plugin(make_napari_viewer_proxy):
    view = make_napari_viewer_proxy()
    widget = Utilities(view)

    image = rand_gen.random((10, 10, 10)).astype(np.uint8)
    image_layer = view.add_image(image, name="image")
    label_layer = view.add_labels(image.astype(np.uint8), name="labels")

    view.window.add_dock_widget(widget)
    view.dims.ndisplay = 3
    for i, utils_name in enumerate(UTILITIES_WIDGETS.keys()):
        widget.utils_choice.setCurrentIndex(i)
        assert isinstance(
            widget.utils_widgets[i], UTILITIES_WIDGETS[utils_name]
        )
        if utils_name == "Convert to instance labels":
            # to avoid issues with Voronoi-Otsu missing runtime
            menu = widget.utils_widgets[i].instance_widgets.method_choice
            menu.setCurrentIndex(menu.currentIndex() + 1)

        assert len(image_layer.data.shape) == 3
        assert len(label_layer.data.shape) == 3
        widget.utils_widgets[i]._start()


def test_crop_widget(make_napari_viewer_proxy):
    view = make_napari_viewer_proxy()
    widget = Cropping(view)

    image = rand_gen.random((10, 10, 10)).astype(np.uint8)
    image_layer_1 = view.add_image(image, name="image")
    image_layer_2 = view.add_labels(image, name="image2")

    view.window.add_dock_widget(widget)
    view.dims.ndisplay = 3
    assert len(image_layer_1.data.shape) == 3
    assert len(image_layer_2.data.shape) == 3
    widget.crop_second_image_choice.setChecked(True)
    widget.aniso_widgets.checkbox.setChecked(True)

    widget._start()
    widget.create_new_layer.setChecked(True)
    widget.quicksave()

    widget.sliders[0].setValue(2)
    widget.sliders[1].setValue(2)
    widget.sliders[2].setValue(2)

    widget._start()


def test_stats_plugin(make_napari_viewer_proxy):
    view = make_napari_viewer_proxy()
    widget = StatsUtils(view)

    labels = rand_gen.random((10, 10, 10)).astype(np.uint8)
    view.add_labels(labels, name="labels")

    view.window.add_dock_widget(widget)
    widget.csv_name.setText("test.csv")
    widget._start()
