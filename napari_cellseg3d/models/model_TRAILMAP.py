from napari_cellseg3d.models.unet.model import UNet3D


def get_weights_file():
    # return "TMP_TEST_40e.pth"
    return "trailmaptorchpretrained.pth"


def get_net():
    return UNet3D(1, 1)


def get_output(model, input):
    out = model(input)

    return out


def get_validation(model, val_inputs):

    return model(val_inputs)
