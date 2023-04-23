import torch

from napari_cellseg3d.code_models.models.wnet.soft_Ncuts import SoftNCutsLoss
from napari_cellseg3d.config import MODEL_LIST


def test_model_list():
    for model_name in MODEL_LIST.keys():
        dims = 128
        test = MODEL_LIST[model_name](
            input_img_size=[dims, dims, dims],
            in_channels=1,
            out_channels=1,
            dropout_prob=0.3,
        )
        assert isinstance(test, MODEL_LIST[model_name])


def test_soft_ncuts_loss():
    dims = 8
    labels = torch.rand([1, 1, dims, dims, dims])

    loss = SoftNCutsLoss(
        data_shape=[dims, dims, dims],
        device="cpu",
        o_i=4,
        o_x=4,
        radius=2,
    )

    res = loss.forward(labels, labels)
    assert isinstance(res, torch.Tensor)
    # assert res > 0
