from monai.inferers import sliding_window_inference
from monai.networks.nets import VNet
from napari_cellseg3d import utils
import os

modelname = "VNet"
target_dir = os.path.join("models","pretrained")

def get_net():
    return VNet()


def get_weights_file():
    utils.download_model(modelname, target_dir)
    return "VNet_40e.pth"


def get_output(model, input):
    out = model(input)
    return out


def get_validation(model, val_inputs):
    roi_size = (64, 64, 64)
    sw_batch_size = 1
    val_outputs = sliding_window_inference(
        val_inputs, roi_size, sw_batch_size, model, mode="gaussian"
    )
    return val_outputs
