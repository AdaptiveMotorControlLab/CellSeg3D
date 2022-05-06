.. _inference_module_guide:

Inference module guide
=================================

This module allows you to use  pre-trained segmentation algorithms (written in Pytorch) on volumes
to automatically label cells.

Currently, the following pre-trained models are available :

===========   ================================================================================================
Model         Link to original paper
===========   ================================================================================================
VNet          `Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation`_
SegResNet     `3D MRI brain tumor segmentation using autoencoder regularization`_
TRAILMAP      An emulation in Pytorch of the `TRAILMAP project on GitHub`_
===========   ================================================================================================

.. _Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation: https://arxiv.org/pdf/1606.04797.pdf
.. _3D MRI brain tumor segmentation using autoencoder regularization: https://arxiv.org/pdf/1810.11654.pdf
.. _TRAILMAP project on GitHub: https://github.com/AlbertPun/TRAILMAP

Interface and functionalities
--------------------------------

.. image:: ../images/inference_plugin_layout.png
    :align: right

* Loading data : When launching the module, you will be asked to provide an image folder containing all the volumes you'd like to be labeled.
  All images with the chosen (**.tif** or **.tiff** currently supported) extension in this folder will be labeled.
  You can then choose an output folder, where all the results will be saved.



* Model choice : You can then choose one of the selected models above, which will be used for inference.



* Anisotropy :If you want to see your results without anisotropy when you have anisotropic images, you can specify that you have anisotropic data
  and set the resolution of your image in micron, this wil save & show the results without anisotropy.



* Thresholding : You can perform thresholding to binarize your labels, all values beneath the confidence threshold will be set to 0 using this.
  If you wish to use instance segmentation it is recommended to use threshlding.

* Instance segmentatin : You can convert the semantic segmentation into instance labels by using either the watershed or connected components method.
  You can set the probability threshhold from which a pixel is considered as a valid instance, as well as the minimum size in pixels for objects. All smaller objects will be removed.
  Instance labels will be saved (and shown if applicable) separately from other results.

* Viewing results : You can also select whether you'd like to see the results in napari afterwards; by default the first image processed will be displayed,
  but you can choose to display up to ten at once. You can also request to see the originals.




When you are done choosing your parameters, you can press the **Start** button to begin the inference process.
Once it has finished, results will be saved then displayed in napari; each output will be paired with its original.
On the left side, a progress bar and a log will keep you informed on the process.



.. note::
    | The files will be saved using the following format :
    |    ``{original_name}_{model}_{date & time}_pred{id}.file_ext``
    | For example, using a VNet on the third image of a folder, called "somatomotor.tif" will yield the following name :
    |   *somatomotor_VNet_2022_04_06_15_49_42_pred3.tif*


.. hint::
    | **Results** will be displayed using the **twilight shifted** colormap if raw or **turbo** if thresholding has been applied, whereas the **original** image will be shown in the **inferno** colormap.
    | Feel free to change the **colormap** or **contrast** when viewing results to ensure you can properly see the labels.
    | You'll most likely want to use **3D view** and **grid mode** in napari when checking results more broadly.

.. image:: ../images/inference_results_example.png

.. note::
    You can save the log after the worker is finished to easily remember which parameters you ran inference with.

Source code
--------------------------------
* :doc:`../code/plugin_model_inference`
* :doc:`../code/model_framework`
