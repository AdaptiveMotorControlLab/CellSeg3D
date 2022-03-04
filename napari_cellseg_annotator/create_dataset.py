import os

import cv2
import tifffile
import numpy as np

# input_path = "/Users/maximevidal/Documents/3drawdata/visual/volumes/images.tif"
input_path = "/Users/maximevidal/Documents/3drawdata/visual/labels/labels.tif"
original_stack = tifffile.imread(input_path)
print(original_stack.shape)

# output_path = "/Users/maximevidal/Documents/3drawdata/visual/annotator"
output_path = "/Users/maximevidal/Documents/3drawdata/visual/sample_labels_semantic"
os.makedirs(output_path, exist_ok=True)
original_stack[original_stack >= 1] = 1
for i in range(original_stack.shape[0]):

    cv2.imwrite(
        f"{output_path}/{str(i).zfill(4)}.png",
        original_stack[i].astype(np.int16),  # for labels
    )

# from dask_image.imread import imread
#
# x = imread('/Users/vmax/Documents/3d/data/ExpC_TPH2_whole_brain-003.tif')
