import pytest
import numpy as np
from pathlib import Path

from napari_cellseg3d.code_models.workers import WeightsDownloader, InferenceWorker
from napari_cellseg3d.config import MODEL_LIST, PRETRAINED_WEIGHTS_DIR, InferenceWorkerConfig

def test_onnx_inference(make_napari_viewer_proxy):
    downloader = WeightsDownloader()
    downloader.download_weights("WNet_ONNX", "wnet.onnx")

    config = InferenceWorkerConfig()
    config.weights_config.path = str(Path(PRETRAINED_WEIGHTS_DIR).resolve() / "wnet.onnx")
    assert Path(config.weights_config.path).is_file()

    viewer = make_napari_viewer_proxy()
    viewer.add_image(np.random.rand(10, 10, 10))
    assert viewer.layers[0].data.shape == (10, 10, 10)
    config.layer = viewer.layers[0].data

    worker = InferenceWorker(worker_config=config)

    res = next(worker.inference())

