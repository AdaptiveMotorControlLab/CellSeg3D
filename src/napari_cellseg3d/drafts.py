import napari
import numpy as np
from magicgui import magicgui
from napari.types import ImageData
from napari.types import LabelsData


@magicgui(call_button="Run Threshold")
def threshold(image: ImageData, threshold: int = 75) -> LabelsData:
    """Threshold an image and return a mask."""
    return (image > threshold).astype(int)


viewer = napari.view_image(np.random.randint(0, 100, (64, 64)))
viewer.window.add_dock_widget(threshold)
threshold()
