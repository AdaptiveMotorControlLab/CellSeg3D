"""SwinUNetR wrapper for napari_cellseg3d."""

import inspect

from monai.networks.nets import SwinUNETR

from napari_cellseg3d.utils import LOGGER

logger = LOGGER


class SwinUNETR_(SwinUNETR):
    """SwinUNETR wrapper for napari_cellseg3d."""

    weights_file = "SwinUNetR_latest.pth"
    default_threshold = 0.4

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
        parent_init = super().__init__
        sig = inspect.signature(parent_init)
        init_kwargs = dict(
            in_channels=in_channels,
            out_channels=out_channels,
            use_checkpoint=use_checkpoint,
            feature_size=48,
            drop_rate=0.5,
            attn_drop_rate=0.5,
            use_v2=True,
            **kwargs,
        )
        if "img_size" in sig.parameters:
            # since MONAI API changes depending on py3.8 or py3.9
            init_kwargs["img_size"] = input_img_size
        if "dropout_prob" in kwargs:
            init_kwargs["drop_rate"] = kwargs["dropout_prob"]
            init_kwargs.pop("dropout_prob")
        try:
            parent_init(**init_kwargs)
        except TypeError as e:
            if "img_size" in init_kwargs:
                logger.warning(
                    "Retrying SwinUNETR initialization without img_size due to "
                    f"MONAI API compatibility issue: {e}"
                )
                init_kwargs.pop("img_size", None)
                parent_init(**init_kwargs)
            else:
                raise

    # def forward(self, x_in):
    #     y = super().forward(x_in)
    # return softmax(y, dim=1)
    # return sigmoid(y)

    # def get_output(self, input):
    #     out = self(input)
    #     return torch.sigmoid(out)

    # def get_validation(self, val_inputs):
    #     return self(val_inputs)
