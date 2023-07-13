from pathlib import Path

import torch
from numpy.random import PCG64, Generator

from napari_cellseg3d.code_models.workers import (
    ONNXModelWrapper,
    WeightsDownloader,
)
from napari_cellseg3d.config import (
    PRETRAINED_WEIGHTS_DIR,
)

rand_gen = Generator(PCG64(12345))


def test_onnx_inference(make_napari_viewer_proxy):
    downloader = WeightsDownloader()
    downloader.download_weights("WNet_ONNX", "wnet.onnx")
    path = str(Path(PRETRAINED_WEIGHTS_DIR).resolve() / "wnet.onnx")
    assert Path(path).is_file()
    dims = 64
    batch = 2
    x = torch.randn(size=(batch, 1, dims, dims, dims))
    worker = ONNXModelWrapper(file_location=path)
    assert worker.eval() is None
    assert worker.to(device="cpu") is None
    assert worker.forward(x).shape == (batch, 2, dims, dims, dims)
