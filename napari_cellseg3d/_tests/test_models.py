import numpy as np
import torch

from napari_cellseg3d.code_models.crf import CRFWorker, correct_shape_for_crf
from napari_cellseg3d.code_models.models.wnet.soft_Ncuts import SoftNCutsLoss
from napari_cellseg3d.config import MODEL_LIST


def test_correct_shape_for_crf():
    test = np.random.rand(1, 1, 8, 8, 8)
    assert correct_shape_for_crf(test).shape == (1, 8, 8, 8)
    test = np.random.rand(8, 8, 8)
    assert correct_shape_for_crf(test).shape == (1, 8, 8, 8)


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


def test_crf(qtbot):
    dims = 8
    mock_image = np.random.rand(1, dims, dims, dims)
    mock_label = np.random.rand(2, dims, dims, dims)

    crf = CRFWorker(mock_image, mock_label)

    def on_yield(result):
        assert isinstance(result, np.ndarray)
        assert result.shape[-3:] == mock_label.shape[-3:]

    crf.yielded.connect(on_yield)
    crf.start()
    with qtbot.waitSignal(
        signal=crf.finished, timeout=60000, raising=False
    ) as blocker:
        blocker.connect(crf.errored)

    mock_image = mock_image[0]
    mock_label = mock_label[0]

    crf = CRFWorker(mock_image, mock_label)
    crf.yielded.connect(on_yield)
    crf.start()
    with qtbot.waitSignal(
        signal=crf.finished, timeout=60000, raising=False
    ) as blocker:
        blocker.connect(crf.errored)
