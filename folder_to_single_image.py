import os
import tifftools
import numpy as np
from dask_image.imread import imread

seg_path = "/Users/maximevidal/Documents/trailmap/data/testing/seg-visual1"
out_path = "/Users/maximevidal/Documents/trailmap/data/testing/seg-visual1-single"

os.makedirs(out_path, exist_ok=True)

tiff_files_li = []
for ti in sorted(os.listdir(seg_path)):
    if '.tif' in ti:
        tiff_files_li.append(os.path.join(seg_path,ti))

tifftools.tiff_concat(tiff_files_li, f"{out_path}/image.tif", overwrite=True)

# segmentations = []
# for file in sorted(os.listdir(seg_path)):
#     segmentations.append(imread(os.path.join(seg_path, file)))
# y_pred = np.squeeze(np.array(segmentations), axis=1)
