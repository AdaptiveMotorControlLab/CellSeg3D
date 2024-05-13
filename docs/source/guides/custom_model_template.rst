.. _custom_model_guide:

Advanced : Custom models
=============================================

.. warning::
    **WIP** : Adding new models is still a work in progress and will likely not work out of the box, leading to errors.

    Please `file an issue`_ if you would like to add a custom model and we will help you get it working.

To add a custom model, you will need a **.py** file with the following structure to be placed in the *napari_cellseg3d/models* folder::

    class ModelTemplate_(ABC): # replace ABC with your PyTorch model class name
        weights_file = (
            "model_template.pth"  # specify the file name of the weights file only
        ) # download URL goes in pretrained_models.json

        @abstractmethod
        def __init__(
            self, input_image_size, in_channels=1, out_channels=1, **kwargs
        ):
            """Reimplement this as needed; only include input_image_size if necessary. For now only in/out channels = 1 is supported."""
            pass

        @abstractmethod
        def forward(self, x):
            """Reimplement this as needed. Ensure that output is a torch tensor with dims (batch, channels, z, y, x)."""
            pass


.. note::
    **WIP** : Currently you must modify :doc:`model_framework.py <../code/_autosummary/napari_cellseg3d.code_models.model_framework>` as well : import your model class and add it to the ``model_dict`` attribute

.. _file an issue: https://github.com/AdaptiveMotorControlLab/CellSeg3D/issues
