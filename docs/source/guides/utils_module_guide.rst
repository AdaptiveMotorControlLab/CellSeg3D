.. _utils_module_guide:

Utilities 🛠
============

Here you will find a range of tools for image processing and analysis.
See `Usage section <https://adaptivemotorcontrollab.github.io/CellSeg3d/welcome.html#usage>`_ for instructions on launching the plugin.

.. note::
    The utility selection menu is found at the bottom of the plugin window.

You may specify the results directory for saving; afterwards you can run each action on a folder or on the currently selected layer.

Available actions
__________________

1. Crop 3D volumes
------------------
Please refer to :ref:`cropping_module_guide` for a guide on using the cropping utility.

2. Convert to instance labels
-----------------------------
This will convert semantic (binary) labels to instance labels (with a unique ID for each object).
The available methods for this are:

* `Connected Components`_ : simple method that will assign a unique ID to each connected component. Does not work well for touching objects (objects will often be fused).
* `Watershed`_ : method based on topographic maps. Works well for clumped objects and anisotropic volumes depending on the quality of topography; clumed objects may be fused if this is not true.
* `Voronoi-Otsu`_ : method based on Voronoi diagrams and Otsu thresholding. Works well for clumped objects but only for "round" objects.

3. Convert to semantic labels
-----------------------------
Transforms instance labels into 0/1 semantic labels, useful for training purposes.

4. Remove small objects
-----------------------
Input a size threshold (in pixels) to eliminate objects below this size.

5. Resize anisotropic images
----------------------------
Input your microscope's resolution to remove anisotropy in images.

6. Threshold images
-------------------
Removes values beneath a certain threshold.

7. Fragment image
-----------------
Break down large images into smaller cubes, optimal for training.

8. Conditional Random Field (CRF)
---------------------------------

.. note::
    This utility is only available if you have installed the `pydensecrf` package.
    You may install it by using the command ``pip install pydensecrf@git+https://github.com/lucasb-eyer/pydensecrf.git#egg=master``.

| Refines semantic predictions by pairing them with the original image.
| For a list of parameters, see the :doc:`CRF API page<../code/_autosummary/napari_cellseg3d.code_models.crf>`.

9. Labels statistics
------------------------------------------------
| Computes statistics for each object in the image.
| Enter the name of the csv file to save the results, then select your layer or folder of labels to compute the statistics.

.. note::
    Images that are not only integer labels will be ignored.

The available statistics are:

For each object :

* Object volume (pixels)
* :math:`X,Y,Z` coordinates of the centroid
* Sphericity

Global metrics :

* Image size
* Total image volume (pixels)
* Total object (labeled) volume (pixels)
* Filling ratio (fraction of the volume that is labeled)
* The number of labeled objects

.. hint::
    Check the ``notebooks`` folder for examples of plots using the statistics CSV file.

10. Clear large labels
----------------------
| Clears labels that are larger than a given threshold.
| This is useful for removing artifacts that are larger than the objects of interest.

Source code
___________

* :doc:`../code/_autosummary/napari_cellseg3d.code_plugins.plugin_convert`
* :doc:`../code/_autosummary/napari_cellseg3d.code_plugins.plugin_crf`


.. links

.. _Watershed: https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_watershed.html
.. _Connected Components: https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.label
.. _Voronoi-Otsu: https://haesleinhuepf.github.io/BioImageAnalysisNotebooks/20_image_segmentation/11_voronoi_otsu_labeling.html
