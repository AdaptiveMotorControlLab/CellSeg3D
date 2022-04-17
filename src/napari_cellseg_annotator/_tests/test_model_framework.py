import warnings

import pytest
import torch

from napari_cellseg_annotator import model_framework


def test_get_padding_dim(make_napari_viewer):
    view = make_napari_viewer()
    tensor = torch.randn(100, 30, 40)
    size = tensor.size()
    widget = model_framework.ModelFramework(view)

    pad = widget.get_padding_dim(size)

    assert pad == [128, 32, 64]

    tensor = torch.randn(2000, 30, 40)
    size = tensor.size()

    warn = warnings.warn(
        "Warning : a very large dimension for automatic padding has been computed.\n"
        "Ensure your images are of an appropriate size and/or that you have enough memory."
        "The padding value is currently 2048."
    )

    pad = widget.get_padding_dim(size)

    pytest.warns(warn, (lambda: widget.get_padding_dim(size)))

    assert pad == [2048, 32, 64]


def test_update_default(make_napari_viewer):
    view = make_napari_viewer()
    widget = model_framework.ModelFramework(view)

    widget.images_filepaths = [""]
    widget.results_path = ""

    widget.update_default()

    assert widget._default_path == []

    widget.images_filepaths = [
        "C:/test/test/images.tif",
        "C:/images/test/data.png",
    ]
    widget.labels_filepaths = [
        "C:/dataset/labels/lab1.tif",
        "C:/data/labels/lab2.tif",
    ]
    widget.results_path = "D:/dataset/res"
    widget.model_path = ""

    widget.update_default()

    assert widget._default_path == [
        "C:/test/test",
        "C:/dataset/labels",
        "D:/dataset/res",
    ]


def test_create_train_dataset_dict(make_napari_viewer):
    view = make_napari_viewer()
    widget = model_framework.ModelFramework(view)

    widget.images_filepaths = [str(f"{i}.tif") for i in range(3)]
    widget.labels_filepaths = [str(f"lab_{i}.tif") for i in range(3)]

    expect = [
        {"image": "0.tif", "label": "lab_0.tif"},
        {"image": "1.tif", "label": "lab_1.tif"},
        {"image": "2.tif", "label": "lab_2.tif"},
    ]

    assert widget.create_train_dataset_dict() == expect
