from pathlib import Path

from tifffile import imread

from napari_cellseg3d.code_plugins.plugin_review_dock import Datamanager


def test_prepare(make_napari_viewer):
    path_image = str(Path(__file__).resolve().parent / "res/test.tif")
    image = imread(str(path_image))
    viewer = make_napari_viewer()
    viewer.add_image(image)
    widget = Datamanager(viewer)

    widget.prepare(path_image, ".tif", "", False)

    assert widget.filetype == ".tif"
    assert widget.as_folder == False
    assert Path(widget.csv_path) == (
        Path(__file__).resolve().parent / "res/_train0.csv"
    )
