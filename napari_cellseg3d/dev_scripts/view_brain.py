import napari
from tifffile import imread

y = imread("/Users/maximevidal/Documents/3drawdata/wholebrain.tif")

with napari.gui_qt():
    viewer = napari.Viewer()
    viewer.add_image(y, contrast_limits=[0, 2000], multiscale=False)
