"""Wrapper for the W-Net model, with the decoder weights removed.

.. important:: Used for inference only. For training the base class is used.
"""

import pickle
from pathlib import Path
from typing import Optional

import torch

# local
from napari_cellseg3d.code_models.models.wnet.model import WNet_encoder
from napari_cellseg3d.utils import LOGGER as logger
from napari_cellseg3d.utils import remap_image

NUM_CLASSES_KEY = "encoder.out_b.module.8.bias"
"""State-dict key of the encoder's final conv bias, whose length is the number of classes."""


class WNet_(WNet_encoder):
    """W-Net wrapper for napari_cellseg3d.

    ..important:: Used for inference only, therefore only the encoder is used. For training the base class is used.
    """

    weights_file = "wnet_latest.pth"
    default_threshold = 0.6
    default_channel = 1
    """Foreground channel of the pretrained weights."""

    def __init__(
        self,
        in_channels=1,
        out_channels=2,
        # num_classes=2,
        **kwargs,
    ):
        """Create a W-Net model.

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels.
            **kwargs: additional arguments to WNet_encoder.
        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            # num_classes=num_classes,
            softmax=False,
        )

    # def train(self: T, mode: bool = True) -> T:
    #     raise NotImplementedError("Training not implemented for WNet")

    def forward(self, x):
        """Forward pass of the W-Net model."""
        norm_x = remap_image(x)
        return super().forward(norm_x)

    @staticmethod
    def num_classes_from_weights(weights_path) -> Optional[int]:
        """Number of encoder output classes stored in a .pth weights file, or None if it cannot be read."""
        if Path(weights_path).suffix != ".pth":
            return None
        try:
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            return int(state_dict[NUM_CLASSES_KEY].shape[0])
        except (
            OSError,
            RuntimeError,
            KeyError,
            AttributeError,
            pickle.UnpicklingError,
        ) as e:
            logger.debug(f"Could not read number of classes from {weights_path}: {e}")
            return None

    def load_state_dict(self, state_dict, strict=True):
        """Load the model state dict for inference, without the decoder weights."""
        encoder_checkpoint = state_dict.copy()
        for k in state_dict:
            if k.startswith("decoder"):
                encoder_checkpoint.pop(k)
        # print(encoder_checkpoint.keys())
        super().load_state_dict(encoder_checkpoint, strict=strict)
