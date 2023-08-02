import torch
from qtpy.QtWidgets import QTextEdit

from napari_cellseg3d.utils import LOGGER as logger


class LogFixture(QTextEdit):
    """Fixture for testing, replaces napari_cellseg3d.interface.Log in model_workers during testing"""

    def __init__(self):
        super(LogFixture, self).__init__()

    def print_and_log(self, text, printing=None):
        print(text)

    def warn(self, warning):
        logger.warning(warning)

    def error(self, e):
        raise (e)


class WNetFixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mock_conv = torch.nn.Conv3d(1, 1, 1)
        self.mock_conv.requires_grad_(False)

    def forward_encoder(self, x):
        return x

    def forward_decoder(self, x):
        return x

    def forward(self, x):
        return self.forward_encoder(x), self.forward_decoder(x)


class ModelFixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mock_conv = torch.nn.Conv3d(1, 1, 1)
        self.mock_conv.requires_grad_(False)

    def forward(self, x):
        return x


class OptimizerFixture:
    def __init__(self):
        self.param_groups = []
        self.param_groups.append({"lr": 0})

    def zero_grad(self):
        pass

    def step(self, *args):
        pass


class SchedulerFixture:
    def step(self, *args):
        pass


class LossFixture:
    def __call__(self, *args):
        return self

    def backward(self, *args):
        pass

    def item(self):
        return 0

    def detach(self):
        return self
