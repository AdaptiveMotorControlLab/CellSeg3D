from pathlib import Path

from napari_cellseg3d import config
from napari_cellseg3d._tests.fixtures import LogFixture
from napari_cellseg3d.code_models.models.model_test import TestModel
from napari_cellseg3d.code_plugins.plugin_model_training import (
    Trainer,
    TrainingReport,
)
from napari_cellseg3d.config import MODEL_LIST


def test_update_loss_plot(make_napari_viewer_proxy):
    view = make_napari_viewer_proxy()
    widget = Trainer(view)

    widget.worker_config = config.TrainingWorkerConfig()
    widget.worker_config.validation_interval = 1
    widget.worker_config.results_path_folder = "."

    epoch_loss_values = [1]
    metric_values = []

    widget.update_loss_plot(epoch_loss_values, metric_values)

    assert widget.dice_metric_plot is None
    assert widget.train_loss_plot is None

    widget.worker_config.validation_interval = 2

    epoch_loss_values = [0, 1]
    metric_values = [0.2]

    widget.update_loss_plot(epoch_loss_values, metric_values)

    assert widget.dice_metric_plot is None
    assert widget.train_loss_plot is None

    epoch_loss_values = [0, 1, 0.5, 0.7]
    metric_values = [0.2, 0.3]

    widget.update_loss_plot(epoch_loss_values, metric_values)

    assert widget.dice_metric_plot is not None
    assert widget.train_loss_plot is not None

    epoch_loss_values = [0, 1, 0.5, 0.7, 0.5, 0.7]
    metric_values = [0.2, 0.3, 0.5, 0.7]

    widget.update_loss_plot(epoch_loss_values, metric_values)

    assert widget.dice_metric_plot is not None
    assert widget.train_loss_plot is not None


def test_training(make_napari_viewer_proxy, qtbot):
    im_path = str(Path(__file__).resolve().parent / "res/test.tif")

    viewer = make_napari_viewer_proxy()
    widget = Trainer(viewer)
    widget.log = LogFixture()
    viewer.window.add_dock_widget(widget)

    widget.images_filepath = None
    widget.labels_filepaths = None

    assert not widget.check_ready()

    widget.images_filepaths = [im_path]
    widget.labels_filepaths = [im_path]
    widget.epoch_choice.setValue(1)
    widget.val_interval_choice.setValue(1)

    assert widget.check_ready()

    #################
    # Training is too long to test properly this way. Do not use on Github
    #################
    MODEL_LIST["test"] = TestModel
    widget.model_choice.addItem("test")
    widget.model_choice.setCurrentText("test")
    worker_config = widget._set_worker_config()
    assert worker_config.model_info.name == "test"
    worker = widget._create_worker_from_config(worker_config)
    worker.config.train_data_dict = [{"image": im_path, "label": im_path}]
    worker.config.val_data_dict = [{"image": im_path, "label": im_path}]
    worker.config.max_epochs = 1
    worker.config.validation_interval = 2
    worker.log_parameters()
    res = next(worker.train())

    assert isinstance(res, TrainingReport)
    assert res.epoch == 1

    widget.worker = worker
    res.show_plot = True
    res.loss_values = [1, 1, 1, 1]
    res.validation_metric = [1, 1, 1, 1]
    widget.on_yield(res)
    assert widget.loss_values == [1, 1, 1, 1]
    assert widget.validation_values == [1, 1, 1, 1]

    # def on_error(e):
    #     print(e)
    #     assert False
    #
    # with qtbot.waitSignal(
    #     signal=widget.worker.finished, timeout=10000, raising=True
    # ) as blocker:
    #     blocker.connect(widget.worker.errored)
    #     widget.worker.error_signal.connect(on_error)
    #     widget.worker.train()
    #     assert widget.worker is not None
