import torch
from qtpy.QtWidgets import QTextEdit

from napari_cellseg3d.utils import LOGGER as logger


class LogFixture(QTextEdit):
    """Fixture for testing, replaces napari_cellseg3d.interface.Log in model_workers during testing."""

    def __init__(self):
        super(LogFixture, self).__init__()

    def print_and_log(self, text, printing=None):
        print(text)

    def warn(self, warning):
        logger.warning(warning)

    def error(self, e):
        raise (e)


class WNetFixture(torch.nn.Module):
    """Fixture for testing, replaces napari_cellseg3d.models.WNet during testing."""

    def __init__(self):
        super().__init__()
        self.mock_conv = torch.nn.Conv3d(1, 1, 1)
        self.mock_conv.requires_grad_(False)

    def forward_encoder(self, x):
        """Forward pass through encoder."""
        return x

    def forward_decoder(self, x):
        """Forward pass through decoder."""
        return x

    def forward(self, x):
        """Forward pass through WNet."""
        return self.forward_encoder(x), self.forward_decoder(x)


class ModelFixture(torch.nn.Module):
    """Fixture for testing, replaces napari_cellseg3d models during testing."""

    def __init__(self):
        """Fixture for testing, replaces models during testing."""
        super().__init__()
        self.mock_conv = torch.nn.Conv3d(1, 1, 1)
        self.mock_conv.requires_grad_(False)

    def forward(self, x):
        """Forward pass through model."""
        return x


class OptimizerFixture:
    """Fixture for testing, replaces optimizers during testing."""

    def __init__(self):
        self.param_groups = []
        self.param_groups.append({"lr": 0})

    def zero_grad(self):
        """Dummy function for zero_grad."""
        pass

    def step(self, *args):
        """Dummy function for step."""
        pass


class SchedulerFixture:
    """Fixture for testing, replaces schedulers during testing."""

    def step(self, *args):
        """Dummy function for step."""
        pass


class LossFixture:
    """Fixture for testing, replaces losses during testing."""

    def __call__(self, *args):
        """Dummy function for __call__."""
        return self

    def backward(self, *args):
        """Dummy function for backward."""
        pass

    def item(self):
        """Dummy function for item."""
        return 0

    def detach(self):
        """Dummy function for detach."""
        return self
