from pathlib import Path

from napari_cellseg3d import config
from napari_cellseg3d.code_plugins.plugin_model_training import Trainer
from napari_cellseg3d._tests.fixtures import LogFixture


def test_training(make_napari_viewer, qtbot):
    im_path = str(Path(__file__).resolve().parent / "res/test.tif")

    viewer = make_napari_viewer()
    widget = Trainer(viewer)
    widget.log = LogFixture()
    viewer.window.add_dock_widget(widget)

    widget.images_filepath = None
    widget.labels_filepaths = None

    assert not widget.check_ready()

    assert widget.filetype_choice.currentText() == ".tif"

    widget.images_filepaths = [im_path]
    widget.labels_filepaths = [im_path]

    assert widget.check_ready()

    #################
    # Training is too long to test properly this way. Do not use on Github
    #################

    # widget.start()
    # assert widget.worker is not None
    #
    # with qtbot.waitSignal(signal=widget.worker.finished, timeout=60000) as blocker:  # wait only for 60 seconds.
    #     blocker.connect(widget.worker.errored)


def test_update_loss_plot(make_napari_viewer):
    view = make_napari_viewer()
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
