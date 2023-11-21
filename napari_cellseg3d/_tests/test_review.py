from pathlib import Path

from napari_cellseg3d.code_plugins import plugin_review as rev


def test_launch_review(make_napari_viewer_proxy):
    view = make_napari_viewer_proxy()
    widget = rev.Reviewer(view)

    # widget.filetype_choice.setCurrentIndex(0)

    im_path = str(Path(__file__).resolve().parent / "res/test.tif")
    lab_path = str(Path(__file__).resolve().parent / "res/test_labels.tif")

    widget.folder_choice.setChecked(True)
    widget.image_filewidget.text_field = im_path
    widget.labels_filewidget.text_field = lab_path
    widget.results_filewidget.text_field = str(
        Path(__file__).resolve().parent / "res"
    )

    widget.run_review()
    widget._viewer.close()

    assert widget._viewer is not None
