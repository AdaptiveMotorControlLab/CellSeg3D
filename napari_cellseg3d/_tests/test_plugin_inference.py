from pathlib import Path

from tifffile import imread

from napari_cellseg3d._tests.fixtures import LogFixture
from napari_cellseg3d.code_models.instance_segmentation import (
    INSTANCE_SEGMENTATION_METHOD_LIST,
)
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

    widget.model_choice.setCurrentText("WNet")
    widget._restrict_window_size_for_model()
    assert widget.window_infer_box.isChecked()
    assert widget.window_size_choice.currentText() == "64"

    test_model_name = "test"
    MODEL_LIST[test_model_name] = TestModel
    widget.model_choice.addItem(test_model_name)
    widget.model_choice.setCurrentText(test_model_name)

    widget.worker_config = widget._set_worker_config()
    assert widget.worker_config is not None
    assert widget.model_info is not None
    worker = widget._create_worker_from_config(widget.worker_config)

    assert worker.config is not None
    assert worker.config.model_info is not None
    worker.config.layer = viewer.layers[0].data
    worker.config.post_process_config.instance.enabled = True
    worker.config.post_process_config.instance.method = (
        INSTANCE_SEGMENTATION_METHOD_LIST["Watershed"]()
    )

    assert worker.config.layer is not None
    worker.log_parameters()

    res = next(worker.inference())
    assert isinstance(res, InferenceResult)
    assert res.result.shape == (8, 8, 8)

    widget.on_yield(res)
