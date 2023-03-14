from tifffile import imread
from pathlib import Path

from napari_cellseg3d.config import MODEL_LIST
from napari_cellseg3d._tests.fixtures import LogFixture
from napari_cellseg3d.code_plugins.plugin_model_inference import Inferer
from napari_cellseg3d.code_models.models.model_test import TestModel


def test_inference(make_napari_viewer, qtbot):

    im_path = str(Path(__file__).resolve().parent / "res/test.tif")
    image = imread(im_path)

    assert image.shape == (6, 6, 6)

    viewer = make_napari_viewer()
    widget = Inferer(viewer)
    widget.log = LogFixture()
    viewer.window.add_dock_widget(widget)
    viewer.add_image(image)

    assert len(viewer.layers) == 1

    widget.window_infer_box.setChecked(True)
    widget.window_overlap_slider.setValue(0)
    widget.keep_data_on_cpu_box.setChecked(True)

    assert widget.check_ready()

    MODEL_LIST["test"] = TestModel
    widget.model_choice.addItem("test")
    widget.setCurrentIndex(-1)

    # widget.start()  # takes too long on Github Actions
    # assert widget.worker is not None

    # with qtbot.waitSignal(signal=widget.worker.finished, timeout=60000, raising=False) as blocker:
    #     blocker.connect(widget.worker.errored)

    # assert len(viewer.layers) == 2
