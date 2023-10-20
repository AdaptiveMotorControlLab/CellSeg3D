"""This is a template for a model class. It is not used in the plugin, but is here to show how to implement a model class.

Please note that custom model implementations are not fully supported out of the box yet, but might be in the future.
"""
from abc import ABC, abstractmethod


class ModelTemplate_(ABC):
    """Template for a model class. This is not used in the plugin, but is here to show how to implement a model class."""

    weights_file = (
        "model_template.pth"  # specify the file name of the weights file only
    )

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
