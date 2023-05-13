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

    # def get_output(self, input):
    #     out = self(input)
    #     return out

    # def get_validation(self, val_inputs): # FIXME standardize
    #     roi_size = (64, 64, 64)
    #     sw_batch_size = 1
    #     val_outputs = sliding_window_inference(
    #         val_inputs,
    #         roi_size,
    #         sw_batch_size,
    #         self,
    #         # mode="gaussian",
    #         # overlap=0.7,
    #     )
    #     return val_outputs
