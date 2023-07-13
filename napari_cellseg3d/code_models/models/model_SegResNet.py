from monai.networks.nets import SegResNetVAE


class SegResNet_(SegResNetVAE):
    use_default_training = True
    weights_file = "SegResNet.pth"

    def __init__(
        self, input_img_size, out_channels=1, dropout_prob=0.3, **kwargs
    ):
        super().__init__(
            input_img_size,
            out_channels=out_channels,
            dropout_prob=dropout_prob,
        )

    def forward(self, x):
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
