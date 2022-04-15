from napari_cellseg_annotator import plugin_review as rev
import os


def test_launch_review(make_napari_viewer, qtbot):

    view = make_napari_viewer()
    widget = rev.Reviewer(view)

    im_path = os.path.dirname(os.path.realpath(__file__)) + "/res/test.png"

    widget.image_path = im_path
    widget.label_path = im_path
    widget.run_review()

    assert widget._viewer is not None
