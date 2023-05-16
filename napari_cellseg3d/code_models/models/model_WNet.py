# local
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
        **kwargs,
    ):
        super().__init__(
            device=device,
            in_channels=in_channels,
            out_channels=out_channels,
            num_classes=num_classes,
        )

    # def train(self: T, mode: bool = True) -> T:
    #     raise NotImplementedError("Training not implemented for WNet")

    def forward(self, x):
        """Forward ENCODER pass of the W-Net model.
        Done this way to allow inference on the encoder only when called by sliding_window_inference.
        """
        return self.forward_encoder(x)
        # enc = self.forward_encoder(x)
        # return self.forward_decoder(enc)

    def load_state_dict(self, state_dict, strict=False):
        """Load the model state dict for inference, without the decoder weights."""
        encoder_checkpoint = state_dict.copy()
        for k in state_dict:
            if k.startswith("decoder"):
                encoder_checkpoint.pop(k)
        # print(encoder_checkpoint.keys())
        super().load_state_dict(encoder_checkpoint, strict=strict)
