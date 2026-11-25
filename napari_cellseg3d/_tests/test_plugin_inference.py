from napari_cellseg3d._tests.fixtures import LogFixture
from napari_cellseg3d.code_models.instance_segmentation import (
    INSTANCE_SEGMENTATION_METHOD_LIST,
    volume_stats,
)
from napari_cellseg3d.code_models.models.model_test import TestModel
from napari_cellseg3d.code_models.workers_utils import InferenceResult
from napari_cellseg3d.code_plugins.plugin_model_inference import (
    Inferer,
)
from napari_cellseg3d.config import MODEL_LIST
from napari_cellseg3d.utils import rand_gen


def test_inference(make_napari_viewer_proxy, qtbot):
    dims = 6
    image = rand_gen.random(size=(dims, dims, dims))
    # assert image.shape == (dims, dims, dims)

    viewer = make_napari_viewer_proxy()
    widget = Inferer(viewer)
    widget.log = LogFixture()
    viewer.window.add_dock_widget(widget)
    viewer.add_image(image)

    assert len(viewer.layers) == 1

    widget.use_window_choice.setChecked(True)
    widget.window_overlap_slider.setValue(0)
    widget.keep_data_on_cpu_box.setChecked(True)

    assert widget.check_ready()

    widget.model_choice.setCurrentText("WNet3D")
    widget._restrict_window_size_for_model()
    assert widget.use_window_choice.isChecked()
    assert widget.window_size_choice.currentText() == "64"

    test_model_name = "test"
    MODEL_LIST[test_model_name] = TestModel
    widget.model_choice.addItem(test_model_name)
    widget.model_choice.setCurrentText(test_model_name)

    widget.use_window_choice.setChecked(False)
    widget.worker_config = widget._set_worker_config()
    assert widget.worker_config is not None
    assert widget.model_info is not None

    worker = widget._create_worker_from_config(widget.worker_config)
    assert worker.config is not None
    assert worker.config.model_info is not None
    assert worker.config.sliding_window_config.is_enabled() is False
    worker.config.layer = viewer.layers[0]
    worker.config.post_process_config.instance.enabled = True
    worker.config.post_process_config.instance.method = (
        INSTANCE_SEGMENTATION_METHOD_LIST["Watershed"]()
    )

    assert worker.config.layer is not None
    worker.log_parameters()

    res = next(worker.inference())
    assert isinstance(res, InferenceResult)
    assert res.semantic_segmentation.shape == (8, 8, 8)
    assert res.instance_labels.shape == (8, 8, 8)
    widget.on_yield(res)

    mock_image = rand_gen.random(size=(10, 10, 10))
    mock_labels = rand_gen.integers(0, 10, (10, 10, 10))
    mock_results = InferenceResult(
        image_id=0,
        original=mock_image,
        instance_labels=mock_labels,
        crf_results=mock_image,
        stats=[volume_stats(mock_labels)],
        semantic_segmentation=mock_image,
        model_name="test",
    )
    num_layers = len(viewer.layers)
    widget.worker_config.post_process_config.instance.enabled = True
    widget._display_results(mock_results)
    assert len(viewer.layers) == num_layers + 4

    # assert widget.check_ready()
    # widget._setup_worker()
    # # widget.config.show_results = True
    # with qtbot.waitSignal(widget.worker.yielded, timeout=10000) as blocker:
    #     blocker.connect(
    #         widget.worker.errored
    #     )  # Can add other signals to blocker
    #     widget.worker.start()

    assert widget.on_finish()


def test_keep_channel_choice(make_napari_viewer_proxy, tmp_path):
    import torch

    from napari_cellseg3d import interface as ui
    from napari_cellseg3d.code_models.models.wnet.model import WNet

    # fresh widget, not the singleton instance closed by test_inference
    ui.QWidgetSingleton._instances.pop(Inferer, None)
    viewer = make_napari_viewer_proxy()
    widget = Inferer(viewer)

    widget.model_choice.setCurrentText("SegResNet")
    assert widget.keep_channel_choice.isHidden()
    assert widget._set_worker_config().keep_channel is None

    widget.model_choice.setCurrentText("WNet3D")
    assert not widget.keep_channel_choice.isHidden()
    assert widget.channel_choice_container.isHidden()
    assert widget.channel_choice.count() == 2

    widget.keep_channel_choice.setChecked(True)
    assert not widget.channel_choice_container.isHidden()
    widget.channel_choice.setCurrentIndex(1)
    worker_config = widget._set_worker_config()
    assert worker_config.keep_channel == 1
    assert worker_config.model_info.num_classes == 2

    weights = tmp_path / "wnet_3_classes.pth"
    torch.save(WNet(num_classes=3).state_dict(), weights)
    widget.weights_config.path = str(weights)
    widget.custom_weights_choice.setChecked(True)
    assert widget.channel_choice.count() == 3
    assert widget.channel_choice.currentIndex() == 1  # kept when still valid
    widget.channel_choice.setCurrentIndex(2)
    worker_config = widget._set_worker_config()
    assert worker_config.keep_channel == 2
    assert worker_config.model_info.num_classes == 3

    widget.custom_weights_choice.setChecked(False)
    assert widget.channel_choice.count() == 2
    assert widget.channel_choice.currentIndex() == 0  # 2 no longer exists

    widget.keep_channel_choice.setChecked(False)
    assert widget._set_worker_config().keep_channel is None
