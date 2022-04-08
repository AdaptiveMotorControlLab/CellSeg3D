# TODO : remove
# from napari_cellseg_annotator.plugin_predicter import Predicter
# from napari_cellseg_annotator.plugin_trainer import Trainer
from napari_cellseg_annotator.plugin_crop import Cropping
from napari_cellseg_annotator.plugin_helper import Helper
from napari_cellseg_annotator.plugin_loader import Loader
from napari_cellseg_annotator.plugin_model_inference import Inferer
from napari_cellseg_annotator.plugin_model_training import Trainer


def napari_experimental_provide_dock_widget():
    return [
        (Loader, {"name": "Review loader"}),
        (Helper, {"name": "Help/About..."}),
        (Inferer, {"name": "Inference loader"}),
        (Trainer, {"name": "Training loader"}),
        (Cropping, {"name": "Crop utility"}),
    ]
