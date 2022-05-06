import napari
from dask_image.imread import imread

# Visual
x = imread(
    "/Users/maximevidal/Documents/trailmap/data/no-edge-validation/visual-original/volumes/images.tif"
)
y_semantic = imread(
    "/Users/maximevidal/Documents/trailmap/data/testing/seg-visual1-single/image.tif"
)
y_instance = imread(
    "/Users/maximevidal/Documents/trailmap/data/instance-testing/test-visual-5.tiff"
)
y_true = imread(
    "/Users/maximevidal/Documents/3drawdata/visual/labels/labels.tif"
)

# SM
# x = imread("/Users/maximevidal/Documents/trailmap/data/no-edge-validation/validation-original/volumes/c5images.tif")
# y = imread("/Users/maximevidal/Documents/trailmap/data/instance-testing/test1.tiff")
# y_true = imread("/Users/maximevidal/Documents/3drawdata/somatomotor/labels/c5labels.tif")

with napari.gui_qt():
    viewer = napari.view_image(
        x, colormap="inferno", contrast_limits=[200, 1000]
    )
    viewer.add_image(y_semantic, name="semantic_predictions", opacity=0.5)
    viewer.add_labels(y_instance, name="instance_predictions", seed=0.6)
    viewer.add_labels(y_true, name="truth", seed=0.6)
