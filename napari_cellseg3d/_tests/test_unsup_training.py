from pathlib import Path

from napari_cellseg3d import config
from napari_cellseg3d.code_plugins.plugin_model_training import (
    Trainer,
)

def test_unsupervised_worker(make_napari_viewer_proxy):
    im_path = Path(__file__).resolve().parent / "res/test.tif"
    # im_path_str = str(im_path)

    unsup_viewer = make_napari_viewer_proxy()
    widget = Trainer(viewer=unsup_viewer)
    widget.device_choice.setCurrentIndex(0)

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

    widget.images_filepaths = [str(im_path.parent)]
    widget.labels_filepaths = [str(im_path.parent)]
    # widget.unsupervised_eval_data = widget.create_train_dataset_dict()
    worker = widget._create_worker(additional_results_description="TEST_3")
    dataloader, eval_dataloader, data_shape = worker._get_data()
    assert widget.unsupervised_eval_data is not None
    assert eval_dataloader is not None
    assert widget.unsupervised_eval_data[0]["image"] is not None
    assert widget.unsupervised_eval_data[0]["label"] is not None
    assert data_shape == (6, 6, 6)
