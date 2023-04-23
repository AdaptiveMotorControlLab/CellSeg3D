from napari_cellseg3d.code_models.models.wnet.model import WNet


class WNet_(WNet):
    use_default_training = False
    weights_file = "wnet.pth"

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        num_classes=2,
        device="cpu",
        **kwargs
    ):
        super().__init__(
            device=device,
            in_channels=in_channels,
            out_channels=out_channels,
            num_classes=num_classes,
        )

    def forward(self, x):
        """Forward ENCODER pass of the W-Net model.
        Done this way to allow inference on the encoder only when called by sliding_window_inference.
        """
        enc = self.forward_encoder(x)
        # dec = self.forward_decoder(enc)
        return enc
