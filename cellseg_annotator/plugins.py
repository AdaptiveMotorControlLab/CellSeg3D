from annotator import Loader, Trainer, Predicter


def napari_experimental_provide_dock_widget():
    return [
        (Loader, {"name": "File loader"}),
        (Trainer, {"name": "Train"}),
        (Predicter, {"name": "Predict"}),
    ]
