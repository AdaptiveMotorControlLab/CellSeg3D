"""Utilities to improve whole-brain regions segmentation."""
import numpy as np
from skimage.measure import label
from skimage.segmentation import find_boundaries


def extract_continuous_region(image):
    """Extract continuous region from image."""
    image = np.where(image > 0, 1, 0)
    return label(image)


def get_boundaries(image_regions, num_iters=1):
    """Obtain boundaries from image regions."""
    boundaries = np.zeros_like(image_regions)
    label_values = np.unique(image_regions)
    iter_n = 0
    new_labels = image_regions
    while iter_n < num_iters:
        for i in label_values:
            if i == 0:
                continue
            boundary = find_boundaries(new_labels == i)
            boundaries += np.where(boundary > 0, i, 0)
            new_labels = np.where(boundary > 0, 0, new_labels)
            iter_n += 1
    return boundaries


def remove_boundaries_from_segmentation(
    image_segmentation, image_labels=None, image=None, thickness_num_iters=1
):
    """Remove boundaries from segmentation.

    Args:
        image_segmentation (np.ndarray): 3D image segmentation.
        image_labels (np.ndarray): 3D integer labels of image segmentation. Use output from extract_continuous_region.
        image (np.ndarray): Additional 3D image used to extract continuous region.
        thickness_num_iters (int): Number of iterations to remove boundaries. A greater number will remove more boundary pixels.
    """
    if image_labels is None:
        image_regions = extract_continuous_region(image_segmentation)
    elif image is not None:
        image_regions = extract_continuous_region(image)
    else:
        image_regions = image_labels
    boundaries = get_boundaries(image_regions, num_iters=thickness_num_iters)

    seg_in = np.where(image_regions > 0, image_segmentation, 0)
    return np.where(boundaries > 0, 0, seg_in)
