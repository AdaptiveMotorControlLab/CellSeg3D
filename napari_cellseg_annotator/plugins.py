from napari_cellseg_annotator.annotator import Loader, Helper


def napari_experimental_provide_dock_widget():

    return [
        (Loader, {"name": "File loader"}),
        (Trainer, {"name": "Trainer"}),
        (Predicter, {"name": "Predicter"}),
        (Helper, {"name": "Help/About..."}),
    ]

