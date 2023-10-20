"""Model for testing purposes."""
import torch
from torch import nn


class TestModel(nn.Module):
    """For tests only."""

    weights_file = "test.pth"

    def __init__(self, **kwargs):
        """Create a TestModel model."""
        super().__init__()
        self.linear = nn.Linear(8, 8)

    def forward(self, x):
        """Forward pass of the TestModel model."""
        return self.linear(torch.tensor(x, requires_grad=True))

    # def get_output(self, _, input):
    #     return input

    # def get_validation(self, val_inputs):
    #     return val_inputs


if __name__ == "__main__":
    model = TestModel()
    model.train()
    model.zero_grad()
    from napari_cellseg3d.config import PRETRAINED_WEIGHTS_DIR

    torch.save(
        model.state_dict(),
        PRETRAINED_WEIGHTS_DIR + f"/{TestModel.weights_file}",
    )
