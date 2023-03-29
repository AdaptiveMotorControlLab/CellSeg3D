import torch
from torch import nn


def get_weights_file():
    return "test.pth"


class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(torch.tensor(x, requires_grad=True))

    def get_net(self):
        return self

    def get_output(self, _, input):
        return input

    def get_validation(self, val_inputs):
        return val_inputs


# if __name__ == "__main__":
#
#     model = TestModel()
#     model.train()
#     model.zero_grad()
#     from napari_cellseg3d.config import WEIGHTS_DIR
#     torch.save(
#         model.state_dict(),
#         WEIGHTS_DIR + f"/{get_weights_file()}"
#     )
