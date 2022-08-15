import torch
from monai.networks.nets import SwinUNETR


def get_weights_file():
    return "Swin64_best_metric.pth"


def get_net(img_size, use_checkpoint=True):
    return SwinUNETR(
        img_size,
        in_channels=1,
        out_channels=1,
        feature_size=48,
        use_checkpoint=use_checkpoint,
    )


def get_output(model, input):
    out = model(input)
    return torch.sigmoid(out)


def get_validation(model, val_inputs):
    return model(val_inputs)
