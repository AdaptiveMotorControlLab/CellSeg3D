import os

from napari_cellseg3d import utils
from napari_cellseg3d.models.unet.model import UNet3D


def get_weights_file():
    # original model from Liqun Luo lab, transfered to pytorch
    target_dir = utils.download_model("TRAILMAP")
    return os.path.join(target_dir, "TRAILMAP_PyTorch.pth")


def get_net():
    return UNet3D(1, 1)


def get_output(model, input):
    out = model(input)

    return out


def get_validation(model, val_inputs):

    return model(val_inputs)
