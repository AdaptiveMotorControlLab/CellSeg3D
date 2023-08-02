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


def test_worker_configs(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    widget = Trainer(viewer=viewer)
    # test supervised config and worker
    widget.device_choice.setCurrentIndex(0)
    widget.model_choice.setCurrentIndex(0)
    widget._toggle_unsupervised_mode(enabled=False)
    assert widget.model_choice.currentText() == list(MODEL_LIST.keys())[0]
    worker = widget._create_worker(additional_results_description="test")
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
    # test unsupervised config and worker
    widget.model_choice.setCurrentText("WNet")
    widget._toggle_unsupervised_mode(enabled=True)
    default_config = config.WNetTrainingWorkerConfig()
    worker = widget._create_worker(additional_results_description="TEST_1")
    excluded = ["results_path_folder", "sample_size", "weights_info"]
    for attr in dir(default_config):
        if not attr.startswith("__") and attr not in excluded:
            assert getattr(default_config, attr) == getattr(
                worker.config, attr
            )
    widget.unsupervised_images_filewidget.text_field.setText(
        str(im_path.parent)
    )
    widget.data = widget.create_dataset_dict_no_labs()
    worker = widget._create_worker(additional_results_description="TEST_2")
    dataloader, eval_dataloader, data_shape = worker._get_data()
    assert eval_dataloader is None
    assert data_shape == (6, 6, 6)

    widget.images_filepaths = [str(im_path)]
    widget.labels_filepaths = [str(im_path)]
    # widget.unsupervised_eval_data = widget.create_train_dataset_dict()
    worker = widget._create_worker(additional_results_description="TEST_3")
    dataloader, eval_dataloader, data_shape = worker._get_data()
    assert widget.unsupervised_eval_data is not None
    assert eval_dataloader is not None
    assert widget.unsupervised_eval_data[0]["image"] is not None
    assert widget.unsupervised_eval_data[0]["label"] is not None
    assert data_shape == (6, 6, 6)


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

    widget.images_filepath = []
    widget.labels_filepaths = []

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
