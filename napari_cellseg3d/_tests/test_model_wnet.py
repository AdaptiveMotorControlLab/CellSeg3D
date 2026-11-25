import torch

from napari_cellseg3d.code_models.models.model_WNet import WNet_
from napari_cellseg3d.code_models.models.wnet.model import WNet


def test_num_classes_from_weights(tmp_path):
    weights = tmp_path / "wnet_3_classes.pth"
    torch.save(WNet(num_classes=3).state_dict(), weights)
    assert WNet_.num_classes_from_weights(weights) == 3

    model = WNet_(out_channels=3)
    model.load_state_dict(torch.load(weights, weights_only=True), strict=True)


def test_num_classes_from_weights_unreadable(tmp_path):
    assert WNet_.num_classes_from_weights(tmp_path / "missing.pth") is None
    assert WNet_.num_classes_from_weights(tmp_path / "model.onnx") is None
    no_key = tmp_path / "other.pth"
    torch.save({"weight": torch.zeros(1)}, no_key)
    assert WNet_.num_classes_from_weights(no_key) is None
    garbage = tmp_path / "garbage.pth"
    garbage.write_bytes(b"not a checkpoint")
    assert WNet_.num_classes_from_weights(garbage) is None
    pickled_model = tmp_path / "pickled_model.pth"
    torch.save(torch.nn.Conv3d(1, 2, 1), pickled_model)
    assert WNet_.num_classes_from_weights(pickled_model) is None
