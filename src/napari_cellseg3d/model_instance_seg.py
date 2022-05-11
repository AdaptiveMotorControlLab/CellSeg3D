from __future__ import division
from __future__ import print_function

import numpy as np
from skimage.measure import label
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed
from skimage.transform import resize
from tifffile import imread


def binary_connected(
    volume, thres=0.5, thres_small=3, scale_factors=(1.0, 1.0, 1.0)
):
    r"""Convert binary foreground probability maps to instance masks via
    connected-component labeling.

    Args:
        volume (numpy.ndarray): foreground probability of shape :math:`(C, Z, Y, X)`.
        thres (float): threshold of foreground. Default: 0.8
        thres_small (int): size threshold of small objects to remove. Default: 128
        scale_factors (tuple): scale factors for resizing in :math:`(Z, Y, X)` order. Default: (1.0, 1.0, 1.0)
    """
    semantic = np.squeeze(volume)
    foreground = semantic > thres  # int(255 * thres)
    segm = label(foreground)
    segm = remove_small_objects(segm, thres_small)

    if not all(x == 1.0 for x in scale_factors):
        target_size = (
            int(semantic.shape[0] * scale_factors[0]),
            int(semantic.shape[1] * scale_factors[1]),
            int(semantic.shape[2] * scale_factors[2]),
        )
        segm = resize(
            segm,
            target_size,
            order=0,
            anti_aliasing=False,
            preserve_range=True,
        )

    return segm


def binary_watershed(
    volume,
    thres_seeding=0.9,
    thres_small=10,
    thres_objects=0.3,
    scale_factors=(1.0, 1.0, 1.0),
    rem_seed_thres=3,
):
    r"""Convert binary foreground probability maps to instance masks via
    watershed segmentation algorithm.

    Note:
        This function uses the `skimage.segmentation.watershed <https://github.com/scikit-image/scikit-image/blob/master/skimage/segmentation/_watershed.py#L89>`_
        function that converts the input image into ``np.float64`` data type for processing. Therefore please make sure enough memory is allocated when handling large arrays.

    Args:
        volume (numpy.ndarray): foreground probability of shape :math:`(C, Z, Y, X)`.
        thres_seeding (float): threshold for seeding. Default: 0.98
        thres_objects (float): threshold for foreground objects. Default: 0.3
        thres_small (int): size threshold of small objects removal. Default: 10
        scale_factors (tuple): scale factors for resizing in :math:`(Z, Y, X)` order. Default: (1.0, 1.0, 1.0)
        rem_seed_thres (int): threshold for small seeds removal. Default : 3
    """
    semantic = np.squeeze(volume)
    seed_map = semantic > thres_seeding
    foreground = semantic > thres_objects
    seed = label(seed_map)
    seed = remove_small_objects(seed, rem_seed_thres)
    segm = watershed(-semantic.astype(np.float64), seed, mask=foreground)
    segm = remove_small_objects(segm, thres_small)

    if not all(x == 1.0 for x in scale_factors):
        target_size = (
            int(semantic.shape[0] * scale_factors[0]),
            int(semantic.shape[1] * scale_factors[1]),
            int(semantic.shape[2] * scale_factors[2]),
        )
        segm = resize(
            segm,
            target_size,
            order=0,
            anti_aliasing=False,
            preserve_range=True,
        )

    return np.array(segm)


def clear_small_objects(image, threshold, is_file_path=False):
    """Calls skimage.remove_small_objects to remove small fragments that might be artifacts.

    Args:
        image: array containing the image
        threshold:  size threshold for removal of objects in pixels. E.g. if 10, all objects smaller than 10 pixels as a whole will be removed.
        is_file_path: if True, will load the image from a file path directly. Default : False

    Returns:
        array: The image with small objects removed
    """

    if is_file_path:
        image = imread(image)

    print(threshold)

    labeled = label(image)

    result = remove_small_objects(labeled, threshold)

    print(np.sum(labeled))
    print(np.sum(result))

    if np.sum(labeled) == np.sum(result):
        print("Warning : no objects were removed")

    if np.amax(image) == 1:
        result = to_semantic(result)

    return result


def to_instance(image, is_file_path=False):
    """Converts a **ground-truth** label to instance (unique id per object) labels. Does not remove small objects.

    Args:
        image: image or path to image
        is_file_path: if True, will consider ``image`` to be a string containing a path to a file, if not treats it as an image data array.

    Returns: resulting converted labels

    """
    if is_file_path:
        image = [imread(image)]
        # image = image.compute()

    result = binary_watershed(
        image, thres_small=0, thres_seeding=0.3, rem_seed_thres=0
    )  # TODO add params

    return result


def to_semantic(image, is_file_path=False):
    """Converts a **ground-truth** label to semantic (binary 0/1) labels.

    Args:
        image: image or path to image
        is_file_path: if True, will consider ``image`` to be a string containing a path to a file, if not treats it as an image data array.

    Returns: resulting converted labels

    """
    if is_file_path:
        image = imread(image)
        # image = image.compute()

    image[image >= 1] = 1
    result = image.astype(np.uint16)
    return result
