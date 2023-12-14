"""Test script for sliding window Voronoi-Otsu segmentation.""."""
import numpy as np
import pyclesperanto_prototype as cle
from tqdm import tqdm


def sliding_window_voronoi_otsu(volume, spot_sigma, outline_sigma, patch_size):
    """Given a volume of dimensions HxWxD, a spot_sigma and an outline_sigma, perform Voronoi-Otsu segmentation on the volume using a sliding window of size patch_size.

    If the edge has been reached, the patch size is reduced
    to fit the remaining space. The result is a segmentation of the same size
    as the input volume.

    Args:
        volume (np.array): The volume to segment.
        spot_sigma (float): The sigma for the spot detection.
        outline_sigma (float): The sigma for the outline detection.
        patch_size (int): The size of the sliding window.
    """
    result = np.zeros(volume.shape, dtype=np.uint32)
    max_label_id = 0
    x, y, z = volume.shape[-3:]
    for i in tqdm(range(0, x, patch_size)):
        for j in range(0, y, patch_size):
            for k in range(0, z, patch_size):
                patch = volume[
                    i : min(i + patch_size, x),
                    j : min(j + patch_size, y),
                    k : min(k + patch_size, z),
                ]
                patch_result = cle.voronoi_otsu_labeling(
                    patch, spot_sigma=spot_sigma, outline_sigma=outline_sigma
                )
                patch_result = np.array(patch_result)
                # make sure labels are unique, only where result is not 0
                patch_result[patch_result > 0] += max_label_id
                result[
                    i : min(i + patch_size, x),
                    j : min(j + patch_size, y),
                    k : min(k + patch_size, z),
                ] = patch_result
                max_label_id = np.max(patch_result)
    return result


# if __name__ == "__main__":
#     import napari
#
#     rand_array = np.random.random((525, 621, 400))
#     rand_array = rand_array > 0.999
#
#     result = sliding_window_voronoi_otsu(rand_array, 0.1, 0.1, 128)
#
#     viewer = napari.Viewer()
#     viewer.add_image(rand_array)
#     viewer.add_labels(result)
#     napari.run()
