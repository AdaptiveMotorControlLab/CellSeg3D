import os

import numpy as np

from napari_cellseg_annotator import utils


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
