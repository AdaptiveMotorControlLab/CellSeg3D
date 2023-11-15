"""SegResNet wrapper for napari_cellseg3d."""
from monai.networks.nets import SegResNetVAE


class SegResNet_(SegResNetVAE):
    """SegResNet_ wrapper for napari_cellseg3d."""

    weights_file = "SegResNet_latest.pth"

    def __init__(
        self, input_img_size, out_channels=1, dropout_prob=0.3, **kwargs
    ):
        """Create a SegResNet model.

        Args:
        input_img_size (tuple): input image size
        out_channels (int): number of output channels
        dropout_prob (float): dropout probability.
        **kwargs: additional arguments to SegResNetVAE.
        """
        super().__init__(
            input_img_size,
            out_channels=out_channels,
            dropout_prob=dropout_prob,
        )

    def forward(self, x):
        """Forward pass of the SegResNet model."""
        res = SegResNetVAE.forward(self, x)
        # logger.debug(f"SegResNetVAE.forward: {res[0].shape}")
        return res[0]

    # def get_model_test(self, size):
    #     return SegResNetVAE(
    #         size, in_channels=1, out_channels=1, dropout_prob=0.3
    #     )

    # def get_output(model, input):
    #     out = model(input)[0]
    #     return out

    # def get_validation(model, val_inputs):
    #     val_outputs = model(val_inputs)
    #     return val_outputs[0]
