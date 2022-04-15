from napari_cellseg_annotator.plugin_crop import Cropping
from napari_cellseg_annotator.plugin_helper import Helper
from napari_cellseg_annotator.plugin_review import Reviewer
from napari_cellseg_annotator.plugin_model_inference import Inferer
from napari_cellseg_annotator.plugin_model_training import Trainer


def napari_experimental_provide_dock_widget():
    return [
        (Reviewer, {"name": "Review loader"}),
        (Helper, {"name": "Help/About..."}),
        (Inferer, {"name": "Inference loader"}),
        (Trainer, {"name": "Training loader"}),
        (Cropping, {"name": "Crop utility"}),
    ]
