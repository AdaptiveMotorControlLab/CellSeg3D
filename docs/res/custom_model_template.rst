.. _custom_model_guide:

Advanced : Declaring a custom model
=============================================

To add a custom model, you will need a **.py** file with the following structure to be placed in the *src/napari_cellseg_annotator/models* folder:


::

    def get_net():
        return ModelClass # should return the class of the model,
        # for example SegResNet or UNET


    def get_weights_file():
        return "weights_file.pth" # name of the weights file for the model,
        # which should be in *src/napari_cellseg_annotator/models/saved_weights*


    def get_output(model, input):
        out = model(input) # should return the model's output as [C, N, D,H,W]
        # (C: channel, N, batch size, D,H,W : depth, height, width)
        return out


    def get_validation(model, val_inputs):
        val_outputs = model(val_inputs) # should return the proper type for validation
        # with single_window_inference from MONAI
        return val_outputs


    def ModelClass(x1,x2...):
        # your Pytorch model here...
        return results


