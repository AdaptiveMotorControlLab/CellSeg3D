"""Simple script to fragment a 3d image into smaller 3d images of size roi_size."""
from pathlib import Path

import numpy as np
from tifffile import imread, imwrite

from napari_cellseg3d.utils import get_all_matching_files


def crop_3d_image(image, roi_size):
    """Crops a 3d image by extracting all regions of size roi_size.

    If the edge of the array is reached, the cropped region is overlapped with the previous cropped region.
    """
    image_size = image.shape
    cropped_images = []
    for i in range(0, image_size[0], roi_size[0]):
        for j in range(0, image_size[1], roi_size[1]):
            for k in range(0, image_size[2], roi_size[2]):
                if i + roi_size[0] >= image_size[0]:
                    crop_location_i = image_size[0] - roi_size[0]
                else:
                    crop_location_i = i
                if j + roi_size[1] >= image_size[1]:
                    crop_location_j = image_size[1] - roi_size[1]
                else:
                    crop_location_j = j
                if k + roi_size[2] >= image_size[2]:
                    crop_location_k = image_size[2] - roi_size[2]
                else:
                    crop_location_k = k
                cropped_images.append(
                    image[
                        crop_location_i : crop_location_i + roi_size[0],
                        crop_location_j : crop_location_j + roi_size[1],
                        crop_location_k : crop_location_k + roi_size[2],
                    ]
                )
    return cropped_images


if __name__ == "__main__":
    image_path = (
        Path().home()
        # / "Desktop/Code/CELLSEG_BENCHMARK/TPH2_DATA/somatomotor_iso"
        # / "Desktop/Code/CELLSEG_BENCHMARK/TPH2_DATA/somatomotor_iso/labels/semantic"
        / "Desktop/Code/CELLSEG_BENCHMARK/TPH2_mesospim/visual_iso/labels/semantic"
    )
    if not image_path.exists() or not image_path.is_dir():
        raise ValueError(f"Image path {image_path} does not exist")
    image_list = get_all_matching_files(image_path)
    for j in image_list:
        print(j)
        image = imread(str(j))
        # crops = crop_3d_image(image, (64, 64, 64))
        crops = [image]
        # viewer = napari.Viewer()
        if not (image_path / "cropped").exists():
            (image_path / "cropped").mkdir(exist_ok=False)
        for i, im in enumerate(crops):
            print(im.shape)
            # viewer.add_image(im)
            imwrite(
                str(image_path / f"cropped/{j.stem}_{i}_crop.tif"),
                im.astype(np.uint16),
                dtype="uint16",
            )
        # napari.run()
