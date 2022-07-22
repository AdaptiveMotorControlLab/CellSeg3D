from monai.networks.nets import SegResNetVAE


def get_net(input_image_size, dropout_prob=None):
    return SegResNetVAE(
        input_image_size, out_channels=1, dropout_prob=dropout_prob
    )


def get_weights_file():
    return "SegResNet.pth"


def get_output(model, input):
    out = model(input)[0]
    return out


def get_validation(model, val_inputs):
    val_outputs = model(val_inputs)
    return val_outputs[0]
