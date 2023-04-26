import glob
import os

import numpy as np
from tifffile import imread, imwrite

# input_seg_path = "C:/Users/Cyril/Desktop/Proj_bachelor/code/pytorch-test3dunet/cropped_visual/train/lab"
# output_seg_path = "C:/Users/Cyril/Desktop/Proj_bachelor/code/pytorch-test3dunet/cropped_visual/train/lab_sem"

input_seg_path = "C:/Users/Cyril/Desktop/Proj_bachelor/code/cellseg-annotator-test/napari_cellseg3d/models/dataset/labels"
output_seg_path = "C:/Users/Cyril/Desktop/Proj_bachelor/code/cellseg-annotator-test/napari_cellseg3d/models/dataset/lab_sem"

filenames = []
paths = []
filetype = ".tif"
for filename in glob.glob(os.path.join(input_seg_path, "*" + filetype)):
    paths.append(filename)
    filenames.append(os.path.basename(filename))
    # print(os.path.basename(filename))
for file in paths:
    image = imread(file)

    image[image >= 1] = 1
    image = image.astype(np.uint16)

    imwrite(output_seg_path + "/" + os.path.basename(file), image)
