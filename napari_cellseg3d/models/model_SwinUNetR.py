from monai.networks.nets import SwinUNETR


def get_weights_file():
    return ""


def get_net():
    return SwinUNETR


def get_output(model, input):
    out = model(input)
    return out


def get_validation(model, val_inputs):
    return model(val_inputs)
