import os

from monai.networks.nets import SegResNetVAE

from napari_cellseg3d import utils


def get_net():
    return SegResNetVAE


def get_weights_file():
    return "SegResNet.pth"


def get_output(model, input):
    out = model(input)[0]
    return out


def get_validation(model, val_inputs):
    val_outputs = model(val_inputs)
    return val_outputs[0]
