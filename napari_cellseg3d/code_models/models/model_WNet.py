"""Wrapper for the W-Net model, with the decoder weights removed.

.. important:: Used for inference only. For training the base class is used.
"""

# local
from napari_cellseg3d.code_models.models.wnet.model import WNet_encoder
from napari_cellseg3d.utils import remap_image


class WNet_(WNet_encoder):
    """W-Net wrapper for napari_cellseg3d.

    ..important:: Used for inference only, therefore only the encoder is used. For training the base class is used.
    """

    weights_file = "wnet_latest.pth"

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

    def load_state_dict(self, state_dict, strict=True):
        """Load the model state dict for inference, without the decoder weights."""
        encoder_checkpoint = state_dict.copy()
        for k in state_dict:
            if k.startswith("decoder"):
                encoder_checkpoint.pop(k)
        # print(encoder_checkpoint.keys())
        super().load_state_dict(encoder_checkpoint, strict=strict)
