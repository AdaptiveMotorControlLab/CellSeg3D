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
TRAILMAP      An emulation in Pytorch of the `TRAIlMAP project on GitHub`_
===========   ================================================================================================

.. _Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation: https://arxiv.org/pdf/1606.04797.pdf
.. _3D MRI brain tumor segmentation using autoencoder regularization: https://arxiv.org/pdf/1810.11654.pdf
.. _TRAILMAP project on GitHub: https://github.com/AlbertPun/TRAILMAP

Interface and functionalities
--------------------------------

When launching the module, you will be asked to provide an image folder containing all the volumes you'd like to be labeled.
All images with the **.tif** extension in this folder will be labeled.
You can then choose an output folder, where all the results will be saved.

.. note::
    | The files will be saved using the following format :
    |    ``{original_name}_{model}_{date & time}_pred{id}.file_ext``
    | For example, using a VNet on the third image of a folder, called "volume_1.tif" will yield :
    |   *volume_1_VNet_2022_04_06_15_49_42_pred3.tif*

You can also select whether you'd like to see the results in napari afterwards; by default the first image processed will be displayed,
but you can choose to display up to three or ten at once.

When you are done choosing your parameters, you can press the **Start** button to begin the inference process.
Once it has finished, results will be saved then displayed in napari; each ouput will be paired with its original.

.. hint::
    | **Results** will be displayed using the **twilight shifted** colormap, whereas the **original** image will show in the **inferno** colormap.
    | Feel free to change the **colormap** or **contrast** when viewing results to ensure you can properly see the labels.
    | You'll most likely want to use **3D view** and **grid mode** in napari when checking results more broadly.

.. image:: images/inference_results_example.png

Source code
--------------------------------
* :doc:`plugin_model_inference`
* :doc:`model_framework`
