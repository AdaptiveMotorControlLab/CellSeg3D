from monai.networks.nets import SwinUNETR

from napari_cellseg3d.utils import LOGGER

logger = LOGGER

from napari_cellseg3d.utils import LOGGER

class SwinUNETR_(SwinUNETR):
    use_default_training = True
    weights_file = "Swin64_best_metric.pth"

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        input_img_size=128,
        use_checkpoint=True,
        **kwargs,
    ):
        try:
            super().__init__(
                input_img_size,
                in_channels=in_channels,
                out_channels=out_channels,
                feature_size=48,
                use_checkpoint=use_checkpoint,
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
            )

    # def get_output(self, input):
    #     out = self(input)
    #     return torch.sigmoid(out)

    # def get_validation(self, val_inputs):
    #     return self(val_inputs)
