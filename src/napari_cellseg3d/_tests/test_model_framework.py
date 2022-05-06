from napari_cellseg3d import model_framework


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
