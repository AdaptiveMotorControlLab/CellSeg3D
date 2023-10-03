.. _inference_module_guide:

Inference
=================================

.. figure:: ../images/plugin_inference.png
    :align: center

    Layout of the inference module

This module allows you to use pre-trained segmentation algorithms (written in Pytorch) on 3D volumes
to automatically label cells.

.. important::
    Currently, only inference on **3D volumes is supported**. If running on folders, your image folder
    should only contain a set of **3D image files** as **.tif**.
    Otherwise you may run inference on layers in napari. Stacks of 2D files can be loaded as 3D volumes in napari.

Currently, the following pre-trained models are available :

==============   ================================================================================================
Model            Link to original paper
==============   ================================================================================================
SwinUNetR         `Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images`_
SegResNet        `3D MRI brain tumor segmentation using autoencoder regularization`_
WNet             `WNet, A Deep Model for Fully Unsupervised Image Segmentation`_
TRAILMAP_MS       An implementation of the `TRAILMAP project on GitHub`_ using `3DUNet for PyTorch`_
VNet             `Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation`_
==============   ================================================================================================

.. _Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation: https://arxiv.org/pdf/1606.04797.pdf
.. _3D MRI brain tumor segmentation using autoencoder regularization: https://arxiv.org/pdf/1810.11654.pdf
.. _TRAILMAP project on GitHub: https://github.com/AlbertPun/TRAILMAP
.. _3DUnet for Pytorch: https://github.com/wolny/pytorch-3dunet
.. _Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images: https://arxiv.org/abs/2201.01266
.. _WNet, A Deep Model for Fully Unsupervised Image Segmentation: https://arxiv.org/abs/1711.08506

.. note::
    For WNet-specific instruction please refer to  the appropriate section below.


Interface and functionalities
--------------------------------

.. figure:: ../images/inference_plugin_layout.png
    :align: right

    Inference parameters

* **Loading data** :

  | When launching the module, you will be asked to provide an **image layer** or an **image folder** with the 3D volumes you'd like to be labeled.
  | If loading from folder : All images with the chosen extension (**.tif** currently supported) in this folder will be labeled.
  | You can then choose an **output folder**, where all the results will be saved.

* **Model choice** :

  | You can then choose one of the provided **models** above, which will be used for inference.
  | You may also choose to **load custom weights** rather than the pre-trained ones, simply ensure they are **compatible** (e.g. produced from the training module for the same model)
  | If you choose to use SegResNet or SwinUNetR with custom weights, you will have to provide the size of images it was trained on to ensure compatibility. (See note below)

.. note::
    Currently the SegResNet and SwinUNetR models require you to provide the size of the images the model was trained with.
    Provided weights use a size of 64, please leave it on the default value if you're not using custom weights.

* **Inference parameters** :

  * Window inference: You can choose to use inference on the whole image at once, which can yield better performance at the cost of more memory.
    For larger images this is not possible, due to memory limitations.
    For this reason, you can use a specific window size to run inference on smaller chunks one by one, for lower memory usage.
  * Window overlap: You may specify the amount of overlap between windows; this overlap helps improve performance by reducing border effects.
    Recommended values are 0.1-0. for 3D inference.
  * Keep on CPU: You can also choose to keep the dataset in the RAM rather than the VRAM to avoid running out of VRAM if you have several images.
  * Device: You can choose to run inference on the CPU or GPU. If you have a GPU, it is recommended to use it for faster inference.

* **Anisotropy** :

  | If you want to see your results without **anisotropy** when you have anisotropic images,
  | you may specify that you have anisotropic data and set the **resolution of your volume in micron**, this will save and show the results without anisotropy.

* **Thresholding** :

  You can perform thresholding to **binarize your labels**.
  All values beneath the chosen **confidence threshold** will be set to 0.

.. hint::
  It is recommended to first run without thresholding. You can then use the napari contrast limits to find a good threshold value,
  and run inference later with your chosen threshold.

* **Instance segmentation** :

  | You can convert the semantic segmentation into instance labels by using either the `Voronoi-Otsu`_, `Watershed`_ or `Connected Components`_ method, as detailed in :ref:`utils_module_guide`.
  | Instance labels will be saved (and shown if applicable) separately from other results.


.. _Watershed: https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_watershed.html
.. _Connected Components: https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.label
.. _Voronoi-Otsu: https://haesleinhuepf.github.io/BioImageAnalysisNotebooks/20_image_segmentation/11_voronoi_otsu_labeling.html


* **Computing objects statistics** :

  You can choose to compute various stats from the labels and save them to a .csv for later use.
  This includes, for each object :

  * Object volume (pixels)
  * :math:`X,Y,Z` coordinates of the centroid
  * Sphericity

  And more general statistics :

  * Image size
  * Total image volume (pixels)
  * Total object (labeled) volume (pixels)
  * Filling ratio (fraction of the volume that is labeled)
  * The number of labeled objects


* **Display options** :

  If running on a folder, you can choose to display the results in napari.
  If selected, you may choose how many results to display at once, and whether to display the original image alongside the results.

Once you are ready, hit the Start button to begin inference.
The log will dislay relevant information on the process.

.. hint::
    You can save the log after the worker is finished to easily remember which parameters you ran inference with.

A progress bar will also keep you informed on progress, mainly when running jobs on a folder.

.. note::
    Please note that for technical reasons, the log cannot currently display window inference progress.
    The progress bar for window inference will be displayed in the terminal, however.
    We will work on improving this in the future.


Once the job has finished, the semantic segmentation will be saved in the output folder.

| The files will be saved using the following format :
| ``{original_name}_{model}_{date & time}_pred{id}.file_ext``
|
| For example, using a VNet on the third image of a folder, called "somatomotor.tif" :
| *somatomotor_VNet_2022_04_06_15_49_42_pred3.tif*
|
| Instance labels will have the "Instance_seg" prefix appended to the name.

The output will also be shown in napari. If you ran on a folder, only your previously selected amount of results will be shown.

.. hint::
    | Feel free to change the **colormap** or **contrast** when viewing results to ensure you can properly see the labels.
    | You may want to use **3D view** and **grid mode** in napari when checking results more broadly.


Plotting results
--------------------------------

In the ``notebooks`` folder you will find an example of plotting cell statistics using the volume statistics computed by the inference module.
Simply load the .csv file in a notebook and use the provided functions to plot the desired statistics.


Unsupervised model - WNet
--------------------------------

| The WNet model, from the paper `WNet, A Deep Model for Fully Unsupervised Image Segmentation`_, is a fully unsupervised model that can be used to segment images without any labels.
| It clusters pixels based on brightness, and can be used to segment cells in a variety of modalities.
| Its use and available options are similar to the above models, with a few notable differences.

.. important::
    Our provided, pre-trained model should use an input size of 64x64x64. As such, window inference is always enabled
    and set to 64. If you want to use a different size, you will have to train your own model using the options listed in :ref:`training_wnet`.

As previously, it requires 3D .tif images (you can also load a 2D stack as 3D via napari).
For the best inference performance, the model should be retrained on images of the same modality as the ones you want to segment.
Please see :ref:`training_wnet` for more details on how to train your own model.

.. hint::
  The WNet always outputs a background class, which due to the unsupervised nature of the model, may be displayed first, showing a very "full" volume.
  THe plugin will automatically try to show the foreground class, but this might not always succeed.
  Should this occur, **change the currently shown class by using the slider at the bottom of the napari window.**

Source code
--------------------------------
* :doc:`../code/_autosummary/napari_cellseg3d.code_plugins.plugin_model_inference`
* :doc:`../code/_autosummary/napari_cellseg3d.code_models.worker_inference`
* :doc:`../code/_autosummary/napari_cellseg3d.code_models.models`
