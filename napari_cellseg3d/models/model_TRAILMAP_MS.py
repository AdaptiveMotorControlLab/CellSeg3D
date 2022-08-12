from napari_cellseg3d.models.unet.model import UNet3D


def get_weights_file():
    # original model from Liqun Luo lab, transferred to pytorch and trained on mesoSPIM-acquired data (mostly cFOS as of July 2022)
    return "TRAILMAP_MS_best_metric_epoch_26.pth"


def get_net():
    return UNet3D(1, 1)


def get_output(model, input):
    out = model(input)

    return out


def get_validation(model, val_inputs):

    return model(val_inputs)
