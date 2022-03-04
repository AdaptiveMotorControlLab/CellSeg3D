import os

import cv2
import tifffile
import numpy as np
from PIL import Image

# input_path = "/Users/vmax/Documents/3d/Annotation_2021/experimentC_partialannotation/images.tif"
input_path = "/Users/vmax/Documents/3d/data/cropped_ok.tif"  # /Users/vmax/Documents/TRAILMAP/data/validation/validation-original/volumes/c5images.tif"#"/Users/vmax/Documents/3d/Annotation_2021/visual_area/experimentC_sample1/images.tif"
output_path = "/Users/vmax/Documents/TRAILMAP/data/testing/crowded"
img = Image.open(input_path)
images = []
for i in range(img.n_frames):
    img.seek(i)
    slice = np.array(img)
    im = Image.fromarray(slice)
    im.save(f"{output_path}/{str(i).zfill(4)}.tif")


# original_stack = tifffile.imread(input_path)
# print(original_stack.shape)
#
# output_path = "/Users/vmax/Documents/TRAILMAP/data/testing/partialC"
# os.makedirs(output_path, exist_ok=True)
#
# for i in range(original_stack.shape[0]):
#     # cv2.imwrite(
#     #     f"{output_path}/{str(i).zfill(4)}.tif",
#     #     original_stack[i]#.astype(np.int16) #for labels
#     # )
#     im = Image.fromarray(original_stack[i])
#
#     im.save(f"{output_path}/{str(i).zfill(4)}.tif")
#
# # from dask_image.imread import imread
# #
# # x = imread('/Users/vmax/Documents/3d/data/ExpC_TPH2_whole_brain-003.tif')
