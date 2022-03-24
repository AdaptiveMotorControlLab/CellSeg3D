from napari_cellseg_annotator.plugin_helper import Helper
from napari_cellseg_annotator.plugin_loader import Loader
from napari_cellseg_annotator.plugin_predicter import Predicter
from napari_cellseg_annotator.plugin_trainer import Trainer
from napari_cellseg_annotator.plugin_crop import Cropping


def napari_experimental_provide_dock_widget():
    return [
        (Loader, {"name": "File loader"}),
        (Helper, {"name": "Help/About..."}),
        (Trainer, {"name": "Trainer"}),
        (Cropping, {"name": "Crop utility"}),
        (Predicter, {"name": "Predicter"}),
    ]
