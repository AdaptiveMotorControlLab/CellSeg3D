from napari_cellseg3d.code_models.model_workers import (
    WeightsDownloader,
    WEIGHTS_DIR,
)

# DISABLED, causes GitHub actions to freeze
def test_weight_download():
    downloader = WeightsDownloader()
    downloader.download_weights("test", "test.pth")
    result_path = WEIGHTS_DIR / "test.pth"

    assert result_path.is_file()
