"""TRAILMAP model, reimplemented in PyTorch."""
from napari_cellseg3d.code_models.models.unet.model import UNet3D
from napari_cellseg3d.utils import LOGGER as logger


class TRAILMAP_MS_(UNet3D):
    """TRAILMAP_MS wrapper for napari_cellseg3d."""

    weights_file = "TRAILMAP_MS_best_metric.pth"

    # original model from Liqun Luo lab, transferred to pytorch and trained on mesoSPIM-acquired data (mostly TPH2 as of July 2022)

    def __init__(self, in_channels=1, out_channels=1, **kwargs):
        """Create a TRAILMAP_MS model.

        Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels.
        **kwargs: additional arguments to UNet3D.
        """
        try:
            super().__init__(
                in_channels=in_channels, out_channels=out_channels, **kwargs
            )
        except TypeError as e:
            logger.warning(f"Caught TypeError: {e}")
            super().__init__(
                in_channels=in_channels, out_channels=out_channels
            )

    # def get_output(self, input):
    #     out = self(input)

    # return out
    #
    # def get_validation(self, val_inputs):
    #     return self(val_inputs)
