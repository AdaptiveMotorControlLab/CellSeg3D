from __future__ import division
from __future__ import print_function

import os

import numpy as np
from dask_image.imread import imread
from PIL import Image
from skimage.measure import label
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed
from skimage.transform import resize


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
    semantic = volume[0]
    foreground = semantic > int(255 * thres)
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
    thres1=0.9,
    thres2=0.3,
    thres_small=3,
    scale_factors=(1.0, 1.0, 1.0),
    seed_thres=3,
):
    r"""Convert binary foreground probability maps to instance masks via
    watershed segmentation algorithm.

    Note:
        This function uses the `skimage.segmentation.watershed <https://github.com/scikit-image/scikit-image/blob/master/skimage/segmentation/_watershed.py#L89>`_
        function that converts the input image into ``np.float64`` data type for processing. Therefore please make sure enough memory is allocated when handling large arrays.

    Args:
        volume (numpy.ndarray): foreground probability of shape :math:`(C, Z, Y, X)`.
        thres1 (float): threshold of seeds. Default: 0.98
        thres2 (float): threshold of foreground. Default: 0.85
        thres_small (int): size threshold of small objects to remove. Default: 128
        scale_factors (tuple): scale factors for resizing in :math:`(Z, Y, X)` order. Default: (1.0, 1.0, 1.0)
    """
    semantic = volume  # [0]
    seed_map = semantic > thres1
    foreground = semantic > thres2
    seed = label(seed_map)
    seed = remove_small_objects(seed, seed_thres)
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

    return segm


def write_tiff_stack(vol, fname):
    im = Image.fromarray(vol[0])
    ims = []

    for i in range(1, vol.shape[0]):
        ims.append(Image.fromarray(vol[i]))

    im.save(fname, save_all=True, append_images=ims)


# Load segmentation
base_path = os.path.abspath(__file__ + "/..")
seg_path = base_path + "/data/testing/seg-edgevisual1"
segmentations = []
for file in sorted(os.listdir(seg_path)):
    segmentations.append(imread(os.path.join(seg_path, file)))
y_pred = np.squeeze(np.array(segmentations), axis=1)

y_pred[y_pred > 0.9] = 1
y_pred[y_pred <= 0.9] = 0
y_pred = y_pred.astype("uint8")

# Run post process
output_watershed_path = (
    base_path + "/data/testing/instance-segmentation-w.tiff"
)
output_connected_path = (
    base_path + "/data/testing/instance-segmentation-c.tiff"
)

bw_result = binary_watershed(y_pred)
bc_result = binary_connected(y_pred)

# Save instance predictions
write_tiff_stack(bw_result, output_watershed_path)
write_tiff_stack(bc_result, output_connected_path)
