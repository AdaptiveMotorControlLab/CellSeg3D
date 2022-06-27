from napari_cellseg3d.models.unet.model import UNet3D
from napari_cellseg3d import utils
import os

modelname = "TRAILMAP"
target_dir = os.path.join("models","pretrained")

def get_weights_file():
    utils.download_model(modelname, target_dir)
    return "TRAILMAP_PyTorch.pth" #original model from Liqun Luo lab, transfered to pytorch


def get_net():
    return UNet3D(1, 1)


def get_output(model, input):
    out = model(input)

    return out


def get_validation(model, val_inputs):

    return model(val_inputs)
