from pathlib import Path

from napari_cellseg3d import config
from napari_cellseg3d._tests.fixtures import LogFixture
from napari_cellseg3d.code_models.models.model_test import TestModel
from napari_cellseg3d.code_models.workers_utils import TrainingReport
from napari_cellseg3d.code_plugins.plugin_model_training import (
    Trainer,
)
from napari_cellseg3d.config import MODEL_LIST

im_path = Path(__file__).resolve().parent / "res/test.tif"
im_path_str = str(im_path)

def test_create_supervised_worker_from_config(make_napari_viewer_proxy):

    viewer = make_napari_viewer_proxy()
    widget = Trainer(viewer=viewer)
    widget.device_choice.setCurrentIndex(0)
    worker = widget._create_worker()
    default_config = config.SupervisedTrainingWorkerConfig()
    excluded = [
        "results_path_folder",
        "loss_function",
        "model_info",
        "sample_size",
        "weights_info",
    ]
    for attr in dir(default_config):
        if not attr.startswith("__") and attr not in excluded:
            assert getattr(default_config, attr) == getattr(
                worker.config, attr
            )


def test_update_loss_plot(make_napari_viewer_proxy):
    view = make_napari_viewer_proxy()
    widget = Trainer(view)

    widget.worker_config = config.SupervisedTrainingWorkerConfig()
    assert widget._is_current_job_supervised() is True
    widget.worker_config.validation_interval = 1
    widget.worker_config.results_path_folder = "."

    epoch_loss_values = {"loss": [1]}
    metric_values = []
    widget.update_loss_plot(epoch_loss_values, metric_values)
    assert widget.plot_2 is None
    assert widget.plot_1 is None

    widget.worker_config.validation_interval = 2

    epoch_loss_values = {"loss": [0, 1]}
    metric_values = [0.2]
    widget.update_loss_plot(epoch_loss_values, metric_values)
    assert widget.plot_2 is None
    assert widget.plot_1 is None

    epoch_loss_values = {"loss": [0, 1, 0.5, 0.7]}
    metric_values = [0.1, 0.2]
    widget.update_loss_plot(epoch_loss_values, metric_values)
    assert widget.plot_2 is not None
    assert widget.plot_1 is not None

    epoch_loss_values = {"loss": [0, 1, 0.5, 0.7, 0.5, 0.7]}
    metric_values = [0.2, 0.3, 0.5, 0.7]
    widget.update_loss_plot(epoch_loss_values, metric_values)
    assert widget.plot_2 is not None
    assert widget.plot_1 is not None


def test_check_matching_losses():
    plugin = Trainer(None)
    config = plugin._set_worker_config()
    worker = plugin._create_supervised_worker_from_config(config)

    assert plugin.loss_list == list(worker.loss_dict.keys())


def test_training(make_napari_viewer_proxy, qtbot):
    viewer = make_napari_viewer_proxy()
    widget = Trainer(viewer)
    widget.log = LogFixture()
    viewer.window.add_dock_widget(widget)

    widget.images_filepath = None
    widget.labels_filepaths = None

    assert not widget.check_ready()

    widget.images_filepaths = [im_path_str]
    widget.labels_filepaths = [im_path_str]
    widget.epoch_choice.setValue(1)
    widget.val_interval_choice.setValue(1)

    assert widget.check_ready()

    MODEL_LIST["test"] = TestModel
    widget.model_choice.addItem("test")
    widget.model_choice.setCurrentText("test")
    widget.unsupervised_mode = False
    worker_config = widget._set_worker_config()
    assert worker_config.model_info.name == "test"
    worker = widget._create_supervised_worker_from_config(worker_config)
    worker.config.train_data_dict = [
        {"image": im_path_str, "label": im_path_str}
    ]
    worker.config.val_data_dict = [
        {"image": im_path_str, "label": im_path_str}
    ]
    worker.config.max_epochs = 1
    worker.config.validation_interval = 2
    worker.log_parameters()
    res = next(worker.train())

    assert isinstance(res, TrainingReport)
    assert res.epoch == 0

    widget.worker = worker
    res.show_plot = True
    res.loss_1_values = {"loss": [1, 1, 1, 1]}
    res.loss_2_values = [1, 1, 1, 1]
    widget.on_yield(res)
    assert widget.loss_1_values["loss"] == [1, 1, 1, 1]
    assert widget.loss_2_values == [1, 1, 1, 1]
