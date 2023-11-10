from pathlib import Path

from napari_cellseg3d.code_models import model_framework
from napari_cellseg3d.config import MODEL_LIST


def pth(path):
    return str(Path(path))


def test_update_default(make_napari_viewer_proxy):
    view = make_napari_viewer_proxy()
    widget = model_framework.ModelFramework(view)

    widget.images_filepaths = []
    widget.results_path = None

    widget._update_default_paths()

    assert widget._default_path == [None, None, None, None]

    widget.images_filepaths = [
        pth("C:/test/test/images.tif"),
        pth("C:/images/test/data.png"),
    ]
    widget.labels_filepaths = [
        pth("C:/dataset/labels/lab1.tif"),
        pth("C:/data/labels/lab2.tif"),
    ]
    widget.results_path = pth("D:/dataset/res")
    # widget.model_path = None

    widget._update_default_paths()

    assert widget._default_path == [
        pth("C:/test/test"),
        pth("C:/dataset/labels"),
        None,
        pth("D:/dataset/res"),
    ]


def test_create_train_dataset_dict(make_napari_viewer_proxy):
    view = make_napari_viewer_proxy()
    widget = model_framework.ModelFramework(view)

    widget.images_filepaths = [str(f"{i}.tif") for i in range(3)]
    widget.labels_filepaths = [str(f"lab_{i}.tif") for i in range(3)]

    expect = [
        {"image": "0.tif", "label": "lab_0.tif"},
        {"image": "1.tif", "label": "lab_1.tif"},
        {"image": "2.tif", "label": "lab_2.tif"},
    ]

    assert widget.create_train_dataset_dict() == expect


def test_log(make_napari_viewer_proxy):
    mock_test = "test"
    framework = model_framework.ModelFramework(
        viewer=make_napari_viewer_proxy()
    )
    framework.log.print_and_log(mock_test)
    assert len(framework.log.toPlainText()) != 0
    assert framework.log.toPlainText() == "\n" + mock_test

    framework.results_path = str(Path(__file__).resolve().parent / "res")
    framework.save_log(do_timestamp=False)
    log_path = Path(__file__).resolve().parent / "res/Log_report.txt"
    assert log_path.is_file()
    with Path.open(log_path.resolve(), "r") as f:
        assert f.read() == "\n" + mock_test

    # remove log file
    log_path.unlink(missing_ok=False)
    log_path = Path(__file__).resolve().parent / "res/Log_report.txt"
    framework.save_log_to_path(str(log_path.parent), do_timestamp=False)
    assert log_path.is_file()
    with Path.open(log_path.resolve(), "r") as f:
        assert f.read() == "\n" + mock_test
    log_path.unlink(missing_ok=False)


def test_display_elements(make_napari_viewer_proxy):
    framework = model_framework.ModelFramework(
        viewer=make_napari_viewer_proxy()
    )

    framework.display_status_report()
    framework.display_status_report()

    framework.custom_weights_choice.setChecked(False)
    framework._toggle_weights_path()
    assert not framework.weights_filewidget.isVisible()


def test_available_models_retrieval(make_napari_viewer_proxy):
    framework = model_framework.ModelFramework(
        viewer=make_napari_viewer_proxy()
    )
    assert framework.get_available_models() == MODEL_LIST


def test_update_weights_path(make_napari_viewer_proxy):
    framework = model_framework.ModelFramework(
        viewer=make_napari_viewer_proxy()
    )
    assert (
        framework._update_weights_path(framework._default_weights_folder)
        is None
    )
    name = str(Path.home() / "test/weight.pth")
    framework._update_weights_path([name])
    assert framework.weights_config.path == name
    assert framework.weights_filewidget.text_field.text() == name
    assert framework._default_weights_folder == str(Path.home() / "test")
