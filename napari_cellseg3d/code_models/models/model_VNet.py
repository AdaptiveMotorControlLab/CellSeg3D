"""VNet wrapper for napari_cellseg3d."""
from monai.networks.nets import VNet


class VNet_(VNet):
    """VNet wrapper for napari_cellseg3d."""

    weights_file = "VNet_latest.pth"

    def __init__(self, in_channels=1, out_channels=1, **kwargs):
        """Create a VNet model.

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels.
            **kwargs: additional arguments to VNet.
        """
        try:
            super().__init__(
                in_channels=in_channels,
                out_channels=out_channels,
                bias=True,
                **kwargs,
            )
        except TypeError:
            super().__init__(
                in_channels=in_channels, out_channels=out_channels, bias=True
            )
