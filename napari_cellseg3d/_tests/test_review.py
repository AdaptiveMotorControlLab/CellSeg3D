from pathlib import Path

from napari_cellseg3d import plugin_review as rev


def test_launch_review(make_napari_viewer):

    view = make_napari_viewer()
    widget = rev.Reviewer(view)

    # widget.filetype_choice.setCurrentIndex(0)

    im_path = str(Path(__file__).resolve().parent / "res/test.tif")

    widget.folder_choice.setChecked(True)
    widget.image_filewidget.text_field = im_path
    widget.labels_filewidget.text_field = im_path

    widget.run_review()
    widget._viewer.close()

    assert widget._viewer is not None
