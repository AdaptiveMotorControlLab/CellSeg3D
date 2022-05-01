import os
import warnings

import numpy as np
import pytest
import torch

from napari_cellseg_annotator import utils


def test_get_padding_dim(make_napari_viewer):

    tensor = torch.randn(100, 30, 40)
    size = tensor.size()

    pad = utils.get_padding_dim(size)

    assert pad == [128, 32, 64]

    tensor = torch.randn(2000, 30, 40)
    size = tensor.size()

    warn = warnings.warn(
        "Warning : a very large dimension for automatic padding has been computed.\n"
        "Ensure your images are of an appropriate size and/or that you have enough memory."
        "The padding value is currently 2048."
    )

    pad = utils.get_padding_dim(size)

    pytest.warns(warn, (lambda: utils.get_padding_dim(size)))

    assert pad == [2048, 32, 64]


def test_normalize_x():

    test_array = utils.normalize_x(np.array([0, 255, 127.5]))
    expected = np.array([-1, 1, 0])
    assert np.all(test_array == expected)


def test_parse_default_path():

    user_path = os.path.expanduser("~")
    assert utils.parse_default_path([""]) == user_path

    path = ["C:/test/test", "", [""]]
    assert utils.parse_default_path(path) == "C:/test/test"

    path = ["C:/test/test", "", [""], "D:/very/long/path/what/a/bore", ""]
    assert utils.parse_default_path(path) == "D:/very/long/path/what/a/bore"
