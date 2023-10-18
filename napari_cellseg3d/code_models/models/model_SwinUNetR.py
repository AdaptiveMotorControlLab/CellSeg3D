"""SwinUNetR wrapper for napari_cellseg3d."""
from monai.networks.nets import SwinUNETR

from napari_cellseg3d.utils import LOGGER

logger = LOGGER


class SwinUNETR_(SwinUNETR):
    """SwinUNETR wrapper for napari_cellseg3d."""

    weights_file = "SwinUNetR_latest.pth"

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        input_img_size=(64, 64, 64),
        use_checkpoint=True,
        **kwargs,
    ):
        """Create a SwinUNetR model.

        Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        input_img_size (tuple): input image size
        use_checkpoint (bool): whether to use checkpointing during training.
        **kwargs: additional arguments to SwinUNETR.
        """
        try:
            super().__init__(
                input_img_size,
                in_channels=in_channels,
                out_channels=out_channels,
                feature_size=48,
                use_checkpoint=use_checkpoint,
                drop_rate=0.5,
                attn_drop_rate=0.5,
                use_v2=True,
                **kwargs,
            )
        except TypeError as e:
            logger.warning(f"Caught TypeError: {e}")
            super().__init__(
                input_img_size,
                in_channels=1,
                out_channels=1,
                feature_size=48,
                use_checkpoint=use_checkpoint,
                drop_rate=0.5,
                attn_drop_rate=0.5,
                use_v2=True,
            )

    # def forward(self, x_in):
    #     y = super().forward(x_in)
    # return softmax(y, dim=1)
    # return sigmoid(y)

    # def get_output(self, input):
    #     out = self(input)
    #     return torch.sigmoid(out)

    # def get_validation(self, val_inputs):
    #     return self(val_inputs)
