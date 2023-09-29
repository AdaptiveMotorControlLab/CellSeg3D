.. _utils_module_guide:

Other utilities
==================================

This utility will let you convert labels to various different formats.

You will have to specify the results directory for saving; afterwards you can run each action on a folder or on the currently selected layer.

You can :

* Crop 3D volumes :
    Please refer to :ref:`cropping_module_guide` for a guide on using the cropping utility.

* Convert to instance labels :
    This will convert 0/1 semantic labels to instance label, with a unique ID for each object.
    The available methods for this are :

    * Connected components : simple method that will assign a unique ID to each connected component. Does not work well for touching objects (objects will often be fused), works for anisotropic volumes.
    * Watershed : method based on topographic maps. Works well for touching objects and anisotropic volumes; touching objects may be fused.
    * Voronoi-Otsu : method based on Voronoi diagrams. Works well for touching objects but only for isotropic volumes.
* Convert to semantic labels :
    This will convert instance labels with unique IDs per object into 0/1 semantic labels, for example for training.

.. figure:: ../images/converted_labels.png
   :scale: 30 %
   :align: center

   Example of instance labels (left) converted to semantic labels (right)

* Remove small objects :
    You can specify a size threshold in pixels; all objects smaller than this size will be removed in the image.

* Resize anisotropic images :
    Specify the resolution of your microscope to remove anisotropy from images.

.. important:: Does not work for instance labels currently.

* Threshold images :
    Remove all values below a threshold in an image.

* Fragment image :
    Break down a large image into cubes suitable for training.

* Conditional Random Field (CRF):
    Attempts to refine semantic predictions by pairing it with the original image.
    For a list of parameters, see the :doc:`CRF API page<../code/_autosummary/napari_cellseg3d.code_models.crf>`.


Source code
-------------------------------------------------

* :doc:`../code/_autosummary/napari_cellseg3d.code_plugins.plugin_convert`
* :doc:`../code/_autosummary/napari_cellseg3d.code_plugins.plugin_crf`
