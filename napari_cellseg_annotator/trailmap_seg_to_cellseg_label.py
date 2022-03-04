import numpy as np
from PIL import Image
import os
import cv2

# input_path = "/Users/vmax/Documents/3d/Annotation_2021/experimentC_partialannotation/images.tif"
input_seg_path = "/Users/vmax/Documents/TRAILMAP/data/testing/seg-visual1/seg-visual1"
input_vol_path = "/Users/vmax/Documents/TRAILMAP/data/testing/visual1"

output_seg_path = "/Users/vmax/Documents/cellseg-annotator/data/seg-visual1"
output_vol_path = "/Users/vmax/Documents/cellseg-annotator/data/visual1"

os.makedirs(output_seg_path, exist_ok=True)
os.makedirs(output_vol_path, exist_ok=True)

for filename in os.listdir(input_seg_path):
    if filename.endswith(".tif"):
        img = Image.open(os.path.join(input_seg_path, filename))
        image = np.array(img)
        image[image > 0.4] = 1
        image[image <= 0.4] = 0
        cv2.imwrite(
            f"{os.path.join(output_seg_path, filename.split('.')[0])}.png", image
        )

for filename in os.listdir(input_vol_path):
    if filename.endswith(".tif"):
        img = Image.open(os.path.join(input_vol_path, filename))
        image = np.array(img)
        cv2.imwrite(
            f"{os.path.join(output_vol_path, filename.split('.')[0])}.png", image
        )
