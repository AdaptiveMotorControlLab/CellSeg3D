from napari_cellseg3d.code_models.model_workers import WeightsDownloader, WEIGHTS_DIR
from napari_cellseg3d.config import ModelInfo


def test_weight_download():

    info = ModelInfo()

    downloader = WeightsDownloader()

    downloader.download_weights(
        info.name,
        info.get_model().get_weights_file()
    )
    result_path = WEIGHTS_DIR / str(info.get_model().get_weights_file())

    assert result_path.is_file()


