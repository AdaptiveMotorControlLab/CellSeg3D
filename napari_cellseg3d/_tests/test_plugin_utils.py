import numpy as np
from numpy.random import PCG64, Generator

from napari_cellseg3d.code_plugins.plugin_utilities import (
    UTILITIES_WIDGETS,
    Utilities,
)

rand_gen = Generator(PCG64(12345))


def test_utils_plugin(make_napari_viewer):
    view = make_napari_viewer()
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
