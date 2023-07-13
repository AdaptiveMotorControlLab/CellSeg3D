from pathlib import Path

from numpy.random import PCG64, Generator

from napari_cellseg3d.code_models.workers import (
    InferenceWorker,
    WeightsDownloader,
)
from napari_cellseg3d.config import (
    PRETRAINED_WEIGHTS_DIR,
    InferenceWorkerConfig,
)

rand_gen = Generator(PCG64(12345))


def test_onnx_inference(make_napari_viewer_proxy):
    downloader = WeightsDownloader()
    downloader.download_weights("WNet_ONNX", "wnet.onnx")

    config = InferenceWorkerConfig()
    config.weights_config.path = str(
        Path(PRETRAINED_WEIGHTS_DIR).resolve() / "wnet.onnx"
    )
    assert Path(config.weights_config.path).is_file()

    viewer = make_napari_viewer_proxy()
    viewer.add_image(rand_gen.random((10, 10, 10)))
    assert viewer.layers[0].data.shape == (10, 10, 10)
    config.layer = viewer.layers[0].data

    worker = InferenceWorker(worker_config=config)

    next(worker.inference())
