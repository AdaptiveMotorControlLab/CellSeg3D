.. _custom_model_guide:

Advanced : Declaring a custom model
=============================================

To add a custom model, you will need a **.py** file with the following structure to be placed in the *napari_cellseg3d/models* folder:

.. note::
    **WIP** : Currently you must modify :ref:`model_framework.py` as well : import your model class and add it to the ``model_dict`` attribute

::

    def get_net():
        return ModelClass # should return the class of the model,
        # for example SegResNet or UNET


    def get_weights_file():
        return "weights_file.pth" # name of the weights file for the model,
        # which should be in *napari_cellseg3d/models/saved_weights*


    def get_output(model, input):
        out = model(input) # should return the model's output as [C, N, D,H,W]
        # (C: channel, N, batch size, D,H,W : depth, height, width)
        return out


    def get_validation(model, val_inputs):
        val_outputs = model(val_inputs) # should return the proper type for validation
        # with sliding_window_inference from MONAI
        return val_outputs


    def ModelClass(x1,x2...):
        # your Pytorch model here...
        return results # should return as [C, N, D,H,W]


