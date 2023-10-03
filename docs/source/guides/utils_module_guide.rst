.. _utils_module_guide:

Utilities
==================================

Here you will find various utilities for image processing and analysis.

.. note::
    The utility selection menu is found at the bottom of the plugin window.

You may specify the results directory for saving; afterwards you can run each action on a folder or on the currently selected layer.

Available actions
__________________

Crop 3D volumes
----------------------------------
Please refer to :ref:`cropping_module_guide` for a guide on using the cropping utility.

Convert to instance labels
----------------------------------
This will convert semantic (binary) labels to instance labels (with a unique ID for each object).
The available methods for this are :

* `Connected Components`_ : simple method that will assign a unique ID to each connected component. Does not work well for touching objects (objects will often be fused).
* `Watershed`_ : method based on topographic maps. Works well for clumped objects and anisotropic volumes depending on the quality of topography; clumed objects may be fused if this is not true.
* `Voronoi-Otsu`_ : method based on Voronoi diagrams and Otsu thresholding. Works well for clumped objects but only for "round" objects.

Convert to semantic labels
----------------------------------
This will convert instance labels with unique IDs per object into 0/1 semantic labels, for example for training.

Remove small objects
----------------------------------
You can specify a size threshold in pixels; all objects smaller than this size will be removed in the image.

Resize anisotropic images
----------------------------------
Specify the resolution of your microscope to remove anisotropy from images.

Threshold images
----------------------------------
Remove all values below a threshold in an image.

Fragment image
----------------------------------
Break down a large image into cubes suitable for training.

Conditional Random Field (CRF)
----------------------------------
| Attempts to refine semantic predictions by pairing it with the original image.
| For a list of parameters, see the :doc:`CRF API page<../code/_autosummary/napari_cellseg3d.code_models.crf>`.


Source code
__________________

* :doc:`../code/_autosummary/napari_cellseg3d.code_plugins.plugin_convert`
* :doc:`../code/_autosummary/napari_cellseg3d.code_plugins.plugin_crf`


.. links

.. _Watershed: https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_watershed.html
.. _Connected Components: https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.label
.. _Voronoi-Otsu: https://haesleinhuepf.github.io/BioImageAnalysisNotebooks/20_image_segmentation/11_voronoi_otsu_labeling.html
