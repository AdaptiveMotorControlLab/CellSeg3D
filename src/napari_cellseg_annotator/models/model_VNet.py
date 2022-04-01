from monai.networks.nets import VNet


def get_net():

    return VNet()


def get_weights_file():
    return "VNet.pth"
