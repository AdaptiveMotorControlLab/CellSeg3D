import numpy as np
from PIL import Image
import os
import cv2
from dask_image.imread import imread

input_path = "/Users/vmax/Documents/3d/data/ok.tif"
output_path = "/Users/vmax/Documents/3d/data/cropped_ok.tif"


def crop(img):
    return img[620:748, 912:1036, 729:857]


img = imread(input_path)
# cropped_np_img = img.map_blocks(crop, dtype="uint16")
# cropped_np_img = np_img[620:748, 912:1036, 729:857]

cropped_img = img[620:748, 912:1036, 729:857].compute()

im = Image.fromarray(cropped_img[0])
ims = []
for i in range(1, cropped_img.shape[0]):
    ims.append(Image.fromarray(cropped_img[i]))

im.save(output_path, save_all=True, append_images=ims)
