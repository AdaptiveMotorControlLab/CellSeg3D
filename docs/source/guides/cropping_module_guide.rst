.. _cropping_module_guide:

Cropping utility
=================================

This module allows you to crop your volumes and labels dynamically,
by selecting a fixed size volume and moving it around the image.

You can then save the cropped volume and labels directly using napari,
by using the **Quicksave** button,
or by selecting the layer and then using **File -> Save selected layer**,
or simply **CTRL+S** once you have selected the correct layer.


Launching the cropping process
---------------------------------

First, simply pick your images using the layer selection dropdown menu.
If you'd like to crop a second image, e.g. labels, at the same time,
simply check the *"Crop another image simultaneously"* checkbox and
pick the corresponding layer.

You can then choose the size of the cropped volume, which will be constant throughout the process; make sure it is correct beforehand.

You can also opt to correct the anisotropy if your image is anisotropic :
simply set the resolution to the one of your microscope.

.. important::
    This will simply scale the image in the viewer, but saved images will **still be anisotropic.** To resize your image, see :doc:`utils_module_guide`

Once you are ready, you can press **Start** to start the review process.
If you'd like to change the size of the volume, change the parameters as previously to your desired size and hit start again.

Creating new layers
---------------------------------
To "zoom in" your volume, you can use the "Create new layers" checkbox to make a new cropping layer controlled by the sliders
next time you hit Start. This way, you can first select your region of interest by using the tool as described above,
then enable the option, select the cropped region produced before as the input layer, and define a smaller crop size in order to crop within your region of interest.

Interface & functionalities
---------------------------------------------------------------

.. image:: ../images/cropping_process_example.png

Once you have launched the review process, you will have access to three sliders that will let
you **change the position** of the cropped volumes and labels in the x,y and z positions.

.. hint::
    If you **cannot see your cropped volume well**, feel free to change the **colormap** of the image and the cropped
    volume to better see them.
    You may want to change the **opacity** and **contrast thresholds** depending on your image, too.


.. note::
    When you are done you can save the cropped volume and labels directly with the
    **Quicksave** button on the lower left, which will save in the folder you picked the image from, or as
    a separate folder if you loaded a folder as a stack.
    If you want more options (name, format) you can save by selecting the layer and then
    using **File -> Save selected layer**, or simply **CTRL+S** once you have selected the correct layer.

..
    Source code
    -------------------------------------------------

    * :doc:`../code/plugin_crop`
    * :doc:`../code/plugin_base`
