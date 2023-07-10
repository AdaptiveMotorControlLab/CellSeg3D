from monai.networks.nets import VNet


class VNet_(VNet):
    use_default_training = True
    weights_file = "VNet_40e.pth"

    def __init__(self, in_channels=1, out_channels=1, **kwargs):
        try:
            super().__init__(
                in_channels=in_channels, out_channels=out_channels, **kwargs
            )
        except TypeError:
            super().__init__(
                in_channels=in_channels, out_channels=out_channels
            )
