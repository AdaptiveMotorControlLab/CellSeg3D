from pathlib import Path
from tifffile import imread
import numpy as np

from napari_cellseg3d.code_plugins.plugin_utilities import Utilities
from napari_cellseg3d.code_plugins.plugin_utilities import UTILITIES_WIDGETS

rand_gen = Generator(PCG64(12345))


def test_utils_plugin(make_napari_viewer):
    view = make_napari_viewer()
    widget = Utilities(view)

    im_path = str(Path(__file__).resolve().parent / "res/test.tif")
    image = imread(im_path)
    view.add_image(image)
    view.add_labels(image.astype(np.uint8))

    view.window.add_dock_widget(widget)
    view.dims.ndisplay = 3
    for i, utils_name in enumerate(UTILITIES_WIDGETS.keys()):
        widget.utils_choice.setCurrentIndex(i)
        assert isinstance(
            widget.utils_widgets[i], UTILITIES_WIDGETS[utils_name]
        )
        widget.utils_widgets[i]._start()
