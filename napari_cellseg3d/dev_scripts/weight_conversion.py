import collections
import os

import torch

from napari_cellseg3d.models.model_TRAILMAP import get_net
from napari_cellseg3d.models.unet.model import UNet3D

# not sure this actually works when put here


def weight_translate(k, w):
    k = key_translate(k)
    if k.endswith(".weight"):
        if w.dim() == 2:
            w = w.t()
        elif w.dim() == 1:
            pass
        elif w.dim() == 4:
            w = w.permute(3, 2, 0, 1)
        else:
            assert w.dim() == 5
            w = w.permute(4, 3, 0, 1, 2)
    return w


def key_translate(k):
    k = (
        k.replace(
            "conv3d/kernel:0",
            "encoders.0.basic_module.SingleConv1.conv.weight",
        )
        .replace(
            "batch_normalization/gamma:0",
            "encoders.0.basic_module.SingleConv1.batchnorm.weight",
        )
        .replace(
            "batch_normalization/beta:0",
            "encoders.0.basic_module.SingleConv1.batchnorm.bias",
        )
        .replace(
            "conv3d_1/kernel:0",
            "encoders.0.basic_module.SingleConv2.conv.weight",
        )
        .replace(
            "batch_normalization_1/gamma:0",
            "encoders.0.basic_module.SingleConv2.batchnorm.weight",
        )
        .replace(
            "batch_normalization_1/beta:0",
            "encoders.0.basic_module.SingleConv2.batchnorm.bias",
        )
        .replace(
            "conv3d_2/kernel:0",
            "encoders.1.basic_module.SingleConv1.conv.weight",
        )
        .replace(
            "batch_normalization_2/gamma:0",
            "encoders.1.basic_module.SingleConv1.batchnorm.weight",
        )
        .replace(
            "batch_normalization_2/beta:0",
            "encoders.1.basic_module.SingleConv1.batchnorm.bias",
        )
        .replace(
            "conv3d_3/kernel:0",
            "encoders.1.basic_module.SingleConv2.conv.weight",
        )
        .replace(
            "batch_normalization_3/gamma:0",
            "encoders.1.basic_module.SingleConv2.batchnorm.weight",
        )
        .replace(
            "batch_normalization_3/beta:0",
            "encoders.1.basic_module.SingleConv2.batchnorm.bias",
        )
        .replace(
            "conv3d_4/kernel:0",
            "encoders.2.basic_module.SingleConv1.conv.weight",
        )
        .replace(
            "batch_normalization_4/gamma:0",
            "encoders.2.basic_module.SingleConv1.batchnorm.weight",
        )
        .replace(
            "batch_normalization_4/beta:0",
            "encoders.2.basic_module.SingleConv1.batchnorm.bias",
        )
        .replace(
            "conv3d_5/kernel:0",
            "encoders.2.basic_module.SingleConv2.conv.weight",
        )
        .replace(
            "batch_normalization_5/gamma:0",
            "encoders.2.basic_module.SingleConv2.batchnorm.weight",
        )
        .replace(
            "batch_normalization_5/beta:0",
            "encoders.2.basic_module.SingleConv2.batchnorm.bias",
        )
        .replace(
            "conv3d_6/kernel:0",
            "encoders.3.basic_module.SingleConv1.conv.weight",
        )
        .replace(
            "batch_normalization_6/gamma:0",
            "encoders.3.basic_module.SingleConv1.batchnorm.weight",
        )
        .replace(
            "batch_normalization_6/beta:0",
            "encoders.3.basic_module.SingleConv1.batchnorm.bias",
        )
        .replace(
            "conv3d_7/kernel:0",
            "encoders.3.basic_module.SingleConv2.conv.weight",
        )
        .replace(
            "batch_normalization_7/gamma:0",
            "encoders.3.basic_module.SingleConv2.batchnorm.weight",
        )
        .replace(
            "batch_normalization_7/beta:0",
            "encoders.3.basic_module.SingleConv2.batchnorm.bias",
        )
        .replace(
            "conv3d_8/kernel:0",
            "decoders.0.basic_module.SingleConv1.conv.weight",
        )
        .replace(
            "batch_normalization_8/gamma:0",
            "decoders.0.basic_module.SingleConv1.batchnorm.weight",
        )
        .replace(
            "batch_normalization_8/beta:0",
            "decoders.0.basic_module.SingleConv1.batchnorm.bias",
        )
        .replace(
            "conv3d_9/kernel:0",
            "decoders.0.basic_module.SingleConv2.conv.weight",
        )
        .replace(
            "batch_normalization_9/gamma:0",
            "decoders.0.basic_module.SingleConv2.batchnorm.weight",
        )
        .replace(
            "batch_normalization_9/beta:0",
            "decoders.0.basic_module.SingleConv2.batchnorm.bias",
        )
        .replace(
            "conv3d_10/kernel:0",
            "decoders.1.basic_module.SingleConv1.conv.weight",
        )
        .replace(
            "batch_normalization_10/gamma:0",
            "decoders.1.basic_module.SingleConv1.batchnorm.weight",
        )
        .replace(
            "batch_normalization_10/beta:0",
            "decoders.1.basic_module.SingleConv1.batchnorm.bias",
        )
        .replace(
            "conv3d_11/kernel:0",
            "decoders.1.basic_module.SingleConv2.conv.weight",
        )
        .replace(
            "batch_normalization_11/gamma:0",
            "decoders.1.basic_module.SingleConv2.batchnorm.weight",
        )
        .replace(
            "batch_normalization_11/beta:0",
            "decoders.1.basic_module.SingleConv2.batchnorm.bias",
        )
        .replace(
            "conv3d_12/kernel:0",
            "decoders.2.basic_module.SingleConv1.conv.weight",
        )
        .replace(
            "batch_normalization_12/gamma:0",
            "decoders.2.basic_module.SingleConv1.batchnorm.weight",
        )
        .replace(
            "batch_normalization_12/beta:0",
            "decoders.2.basic_module.SingleConv1.batchnorm.bias",
        )
        .replace(
            "conv3d_13/kernel:0",
            "decoders.2.basic_module.SingleConv2.conv.weight",
        )
        .replace(
            "batch_normalization_13/gamma:0",
            "decoders.2.basic_module.SingleConv2.batchnorm.weight",
        )
        .replace(
            "batch_normalization_13/beta:0",
            "decoders.2.basic_module.SingleConv2.batchnorm.bias",
        )
        .replace("conv3d_14/kernel:0", "final_conv.weight")
        .replace("conv3d_14/bias:0", "final_conv.bias")
    )
    return k


model = get_net()
base_path = os.path.abspath(__file__ + "/..")
weights_path = base_path + "/data/model-weights/trailmap_model.hdf5"
model.load_weights(weights_path)

for i, l in enumerate(model.layers):
    print(i, l)
    print(
        "L{}: {}".format(
            i, ", ".join(str(w.shape) for w in model.layers[i].weights)
        )
    )

weights_pt = collections.OrderedDict(
    [(w.name, torch.from_numpy(w.numpy())) for w in model.trainable_variables]
)
torch.save(weights_pt, base_path + "/data/model-weights/trailmaptorch.pt")
torch_weights = torch.load(base_path + "/data/model-weights/trailmaptorch.pt")
param_dict = {
    key_translate(k): weight_translate(k, v) for k, v in torch_weights.items()
}

trailmap_model = UNet3D(1, 1)
torchparam = trailmap_model.state_dict()
for k, v in torchparam.items():
    print("{:20s} {}".format(k, v.shape))

trailmap_model.load_state_dict(param_dict, strict=False)
torch.save(
    trailmap_model.state_dict(),
    base_path + "/data/model-weights/trailmaptorchpretrained.pt",
)
