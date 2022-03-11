from napari_cellseg_annotator.plugin_helper import Helper
from napari_cellseg_annotator.plugin_loader import Loader


def napari_experimental_provide_dock_widget():
    return [
        (Loader, {"name": "File loader"}),
        (Helper, {"name": "Help/About..."}),
        # (Trainer, {"name": "Trainer"}),
        # (Predicter, {"name": "Predicter"}),
    ]
