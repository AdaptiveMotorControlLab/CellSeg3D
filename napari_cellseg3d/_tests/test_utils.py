import os
import warnings

import numpy as np
import pytest
import torch

from napari_cellseg3d import utils


def test_fill_list_in_between():

    list = [1, 2, 3, 4, 5, 6]
    res = [
        1,
        "",
        "",
        2,
        "",
        "",
        3,
        "",
        "",
        4,
        "",
        "",
        5,
        "",
        "",
        6,
        "",
        "",
    ]

    assert utils.fill_list_in_between(list, 2, "") == res


def test_align_array_sizes():

    im = np.zeros((128, 512, 256))
    print(im.shape)

    dim_1 = (64, 64, 512)
    ground = np.array((512, 64, 64))
    pred = np.array(dim_1)

    ori, targ = utils.align_array_sizes(ground, pred)

    im_1 = np.moveaxis(im, ori, targ)
    print(im_1.shape)
    assert im_1.shape == (512, 256, 128)

    dim_2 = (512, 256, 128)
    ground = np.array((128, 512, 256))
    pred = np.array(dim_2)

    ori, targ = utils.align_array_sizes(ground, pred)

    im_2 = np.moveaxis(im, ori, targ)
    print(im_2.shape)
    assert im_2.shape == dim_2

    dim_3 = (128, 128, 128)
    ground = np.array(dim_3)
    pred = np.array(dim_3)

    ori, targ = utils.align_array_sizes(ground, pred)
    im_3 = np.moveaxis(im, ori, targ)
    print(im_3.shape)
    assert im_3.shape == im.shape


def test_get_padding_dim():

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

    tensor = torch.randn(65, 70, 80)
    size = tensor.size()

    pad = utils.get_padding_dim(size)

    assert pad == [128, 128, 128]


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
