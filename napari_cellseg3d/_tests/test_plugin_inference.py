from pathlib import Path

from tifffile import imread

from napari_cellseg3d._tests.fixtures import LogFixture
from napari_cellseg3d.code_models.models.model_test import TestModel
from napari_cellseg3d.code_plugins.plugin_model_inference import (
    InferenceResult,
    Inferer,
)
from napari_cellseg3d.config import MODEL_LIST


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

    widget.worker_config = widget._set_worker_config()
    assert widget.worker_config is not None
    assert widget.model_info is not None
    worker = widget._create_worker_from_config(widget.worker_config)
    assert worker.config is not None
    assert worker.config.model_info is not None
    worker.config.layer = viewer.layers[0].data
    assert worker.config.layer is not None
    worker.log_parameters()

    res = next(worker.inference())
    assert isinstance(res, InferenceResult)
    assert res.result.shape == (6, 6, 6)

    # def on_error(e):
    #     print(e)
    #     assert False
    # with qtbot.waitSignal(
    #     signal=worker.finished, timeout=10000, raising=True
    # ) as blocker:
    #     worker.error_signal.connect(on_error)
    #     blocker.connect(worker.errored)
    #     worker.inference()  # takes too long on Github Actions
    # assert len(viewer.layers) == 2
