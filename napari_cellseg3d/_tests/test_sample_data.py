from napari_cellseg3d import _sample_data


def test_c5image_from_repo():
    assert _sample_data.C5IMAGE_REPO_FILE.is_file()
    layers = _sample_data.c5image()
    assert len(layers) == 1
    data, meta, layer_type = layers[0]
    assert data.shape == (124, 86, 94)
    assert meta["name"] == "c5image"
    assert layer_type == "image"


def test_c5image_open_sample(make_napari_viewer_proxy):
    assert _sample_data.C5IMAGE_REPO_FILE.is_file()
    viewer = make_napari_viewer_proxy()
    viewer.open_sample("napari-cellseg3d", "c5image")
    assert viewer.layers[-1].name == "c5image"
    assert viewer.layers[-1].data.shape == (124, 86, 94)
