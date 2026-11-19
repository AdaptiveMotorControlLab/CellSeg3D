import numpy as np

from napari_cellseg3d.code_models.instance_segmentation import to_semantic


def test_to_semantic_does_not_modify_input():
    labels = np.array([[0, 1, 2], [70_000, 3, 0]], dtype=np.uint32)
    original = labels.copy()

    semantic = to_semantic(labels)

    np.testing.assert_array_equal(labels, original)
    np.testing.assert_array_equal(semantic, [[0, 1, 1], [1, 1, 0]])
    assert semantic.dtype == np.uint16
