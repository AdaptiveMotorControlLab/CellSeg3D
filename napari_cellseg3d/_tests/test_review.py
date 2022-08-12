import os

from napari_cellseg3d import plugin_review as rev


def test_launch_review(make_napari_viewer):

    view = make_napari_viewer()
    widget = rev.Reviewer(view)

    # widget.filetype_choice.setCurrentIndex(0)

    im_path = os.path.dirname(os.path.realpath(__file__)) + "/res/test.tif"

    widget.image_path = im_path
    widget.label_path = im_path

    print(widget.image_path)
    print(widget.label_path)
    print(widget.as_folder)
    print(widget.filetype)
    widget.run_review()
    widget._viewer.close()

    assert widget._viewer is not None
