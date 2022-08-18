from napari_cellseg3d import plugin_model_training as train
from napari_cellseg3d import config


def test_check_ready(make_napari_viewer):
    view = make_napari_viewer()
    widget = train.Trainer(view)

    widget.images_filepath = None
    widget.labels_filepaths = None

    res = widget.check_ready()
    assert not res

    # widget.images_filepath = ["C:/test/something.tif"]
    # widget.labels_filepaths = ["C:/test/lab_something.tif"]
    # res = widget.check_ready()
    #
    # assert res


def test_update_loss_plot(make_napari_viewer):
    view = make_napari_viewer()
    widget = train.Trainer(view)

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
