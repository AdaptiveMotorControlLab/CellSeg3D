import numpy as np
from PIL import Image
import os
import cv2
from dask_image.imread import imread

input_path = "/Users/vmax/Documents/TRAILMAP/data/validation/visual-original/labels/"

for filename in sorted(os.listdir(input_path)):
    if filename.endswith(".tif"):
        print(filename)
        print(len(np.unique(imread(os.path.join(input_path, filename)).compute())))

