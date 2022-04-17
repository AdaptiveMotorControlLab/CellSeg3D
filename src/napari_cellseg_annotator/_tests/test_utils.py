import numpy as np

from napari_cellseg_annotator import utils


def test_normalize_x():

    test_array = utils.normalize_x(np.array([0, 255, 127.5]))
    expected = np.array([-1, 1, 0])
    assert np.all(test_array == expected)
