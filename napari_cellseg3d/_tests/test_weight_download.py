from napari_cellseg3d.code_models.workers_utils import (
    PRETRAINED_WEIGHTS_DIR,
    WeightsDownloader,
)


# DISABLED, causes GitHub actions to freeze
def test_weight_download():
    downloader = WeightsDownloader()
    downloader.download_weights("test", "test.pth")
    result_path = PRETRAINED_WEIGHTS_DIR / "test.pth"

    assert result_path.is_file()
