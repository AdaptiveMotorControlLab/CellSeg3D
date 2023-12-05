from pathlib import Path

import napari
import numpy as np
import pytest
import torch
from monai.data import DataLoader

from napari_cellseg3d.code_models.worker_inference import InferenceWorker
from napari_cellseg3d.code_models.workers_utils import (
    PRETRAINED_WEIGHTS_DIR,
    InferenceResult,
    ONNXModelWrapper,
    WeightsDownloader,
)
from napari_cellseg3d.config import InferenceWorkerConfig
from napari_cellseg3d.utils import rand_gen


def test_onnx_inference(make_napari_viewer_proxy):
    downloader = WeightsDownloader()
    downloader.download_weights("WNet_ONNX", "wnet.onnx")
    path = str(Path(PRETRAINED_WEIGHTS_DIR).resolve() / "wnet.onnx")
    assert Path(path).is_file()
    dims = 64
    batch = 1
    x = torch.randn(size=(batch, 1, dims, dims, dims))
    worker = ONNXModelWrapper(file_location=path)
    assert worker.eval() is None
    assert worker.to(device="cpu") is None
    assert worker.forward(x).shape == (batch, 2, dims, dims, dims)


def test_load_folder():
    config = InferenceWorkerConfig()
    worker = InferenceWorker(worker_config=config)

    images_path = Path(__file__).resolve().parent / "res/test.tif"
    worker.config.images_filepaths = [str(images_path)]
    dataloader = worker.load_folder()
    assert isinstance(dataloader, DataLoader)
    assert len(dataloader) == 1
    worker.config.sliding_window_config.window_size = [64, 64, 64]
    dataloader = worker.load_folder()
    assert isinstance(dataloader, DataLoader)
    assert len(dataloader) == 1

    mock_layer = napari.layers.Image(data=rand_gen.random((64, 64, 64)))
    worker.config.layer = mock_layer
    input_image = worker.load_layer()
    assert input_image.shape == (1, 1, 64, 64, 64)

    mock_layer = napari.layers.Image(data=rand_gen.random((5, 2, 64, 64, 64)))
    worker.config.layer = mock_layer
    assert len(mock_layer.data.shape) == 5
    with pytest.raises(
        ValueError,
        match="Data array is not 3-dimensional but 5-dimensional, please check for extra channel/batch dimensions",
    ):
        worker.load_layer()


def test_inference_on_folder():
    config = InferenceWorkerConfig()
    config.filetype = ".tif"
    config.images_filepaths = [
        str(Path(__file__).resolve().parent / "res/test.tif")
    ]

    config.sliding_window_config.window_size = 8

    class mock_work:
        @staticmethod
        def eval():
            return True

        def __call__(self, x):
            return torch.Tensor(x)

    worker = InferenceWorker(worker_config=config)
    worker.aniso_transform = mock_work()

    image = torch.Tensor(rand_gen.random(size=(1, 1, 8, 8, 8)))
    assert image.shape == (1, 1, 8, 8, 8)
    assert image.dtype == torch.float32
    res = worker.inference_on_folder(
        {"image": image},
        0,
        model=mock_work(),
        post_process_transforms=mock_work(),
    )
    assert isinstance(res, InferenceResult)
    assert res.semantic_segmentation is not None


def test_post_processing():
    image = rand_gen.random((1, 1, 64, 64, 64))
    labels = rand_gen.random((1, 2, 64, 64, 64))
    mock_layer = napari.layers.Image(data=image)
    mock_layer.name = "test"

    config = InferenceWorkerConfig()
    config.layer = mock_layer
    worker = InferenceWorker(worker_config=config)

    results = worker.run_crf(image, labels, lambda x: x)
    assert results.shape == (2, 64, 64, 64)

    worker.stats_csv(np.squeeze(labels))
