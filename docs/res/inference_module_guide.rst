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
===========   ================================================================================================

.. _Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation: https://arxiv.org/pdf/1606.04797.pdf
.. _3D MRI brain tumor segmentation using autoencoder regularization: https://arxiv.org/pdf/1810.11654.pdf

Interface and functionalities
--------------------------------

When launching the module, you will be asked to provide an image folder containing all the volumes you'd like to be labeled.
All images with the **.tif** extension in this folder will be labeled.
You can then choose an output folder, where all the results will be saved.

.. hint::
    | The files will be saved using the following format :
    |    ``{original_name}_{model}_{date & time}_pred{id}.file_ext``
    | For example, using a VNet on the third image of a folder, called "volume_1.tif" will yield :
    |   *volume_1_VNet_2022_04_06_15_49_42_pred3.tif*

You can also select whether you'd like to see the results in napari afterwards; by default the first image processed will be displayed,
but you can choose to display up to three or ten at once.

.. note::
    Feel free to change the **colormap** or **contrast** when viewing results to ensure you can properly see the labels.
    You'll most likely want to use **3D view** and **grid mode** in napari when checking results more broadly.

Source code
--------------------------------
* :doc:`plugin_model_inference`
* :doc:`model_framework`
