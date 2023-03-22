from napari_cellseg3d.code_models.models.unet.model import UNet3D
from napari_cellseg3d.utils import LOGGER

logger = LOGGER


class TRAILMAP_MS_(UNet3D):
    use_default_training = True
    weights_file = "TRAILMAP_MS_best_metric_epoch_26.pth"

    return out


def get_validation(model, val_inputs):
    return model(val_inputs)
