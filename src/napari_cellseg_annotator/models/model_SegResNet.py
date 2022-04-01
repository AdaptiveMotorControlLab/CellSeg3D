from monai.networks.nets import SegResNetVAE


def get_net():

    return SegResNetVAE(
        input_image_size=[128, 128, 128], out_channels=1, dropout_prob=0.1
    )


def get_weights_file():
    return "SegResNet.pth"
