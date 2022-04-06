.. _cropping_module_guide:

Cropping module guide
=================================

This module allows you to crop your volumes and labels dynamically,
by selecting a fixed size volume and moving it around the image.

You can then save the cropped volume and labels directly using napari,
by selecting the layer and then using **File -> Save selected layer**,
or simply **CTRL+S** once you have selected the correct layer.



Launching the cropping process
---------------------------------

First, you will be asked to load your images and labels; you can use the checkbox above the Open buttons to
choose whether you want to load a single 3D **.tif** image or a folder of 2D images as a 3D stack.
Folders can be stacks of either .png or .tif files, ideally numbered with the index of the slice at the end.

.. note::
    Only single .tif files or one folder of several .png or .tif (stack of 2D images) are supported.

You can the choose the size of the cropped volume, which will be constant throughout the process; make sure it is correct beforehand.
Support for dynamically changing the crop size might be added in the future.

Once you are ready, you can press **Start** to start the review process.



Interface & functionalities
---------------------------------------------------------------

.. image:: images/cropping_process_example.png

Once you have launched the review process, you will have access to three sliders that will let
you **change the position** of the cropped volumes and labels in the x,y and z positions.

.. hint::
    If you **cannot see your cropped volume well**, feel free to change the **colormaps** of the image and the cropped
    volume to better see them.
    You may want to change the **opacity** and **contrast thresholds** depending on your image, too.


.. note::
    When you are done you can save the cropped volume and labels directly using napari,
    by selecting the layer and then using **File -> Save selected layer**,
    or simply **CTRL+S** once you have selected the correct layer.




Source code
-------------------------------------------------

* :doc:`plugin_crop`
* :doc:`plugin_base`
