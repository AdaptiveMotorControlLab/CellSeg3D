import torch
from torch import nn


class TestModel(nn.Module):
    use_default_training = True
    weights_file = "test.pth"

    def __init__(self, **kwargs):
        super().__init__()
        self.linear = nn.Linear(8, 8)

    def forward(self, x):
        return self.linear(torch.tensor(x, requires_grad=True))

    # def get_output(self, _, input):
    #     return input

    # def get_validation(self, val_inputs):
    #     return val_inputs


# if __name__ == "__main__":
#
#     model = TestModel()
#     model.train()
#     model.zero_grad()
#     from napari_cellseg3d.config import PRETRAINED_WEIGHTS_DIR
#     torch.save(
#         model.state_dict(),
#         PRETRAINED_WEIGHTS_DIR + f"/{get_weights_file()}"
#     )
