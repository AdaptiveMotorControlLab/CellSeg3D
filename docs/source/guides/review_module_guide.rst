.. _loader_module_guide:

Review
=================================

This module allows you to review your labels, from predictions or manual labeling,
and correct them if needed. It then saves the status of each file in a csv, as well as the time taken per slice review, for easier monitoring.



Launching the review process
---------------------------------

* Data paths :
    First, you will be asked to load your images and labels; you can use the checkbox above the Open buttons to
    choose whether you want to load a single 3D **.tif** image or a folder of 2D images as a 3D stack.
    Folders can be stacks of either **.png** or **.tif** files, ideally numbered with the index of the slice at the end.

.. note::
    Only single 3D **.tif** files or 2D stacks of several **.png** or **.tif** in a folder are currently supported.

* Anisotropic data :
    This will scale the images to visually remove the anisotropy, so as to make review easier.

.. important::
    Results will still be saved as anisotropic images. If you wish to resize your images, see the :doc:`utils_module_guide`

* CSV file name :
    You can then provide a model name, which will be used to name the csv file recording the status of each slice.

    If a corresponding csv file exists already, it will be used. If not, a new one will be created.
    If you choose to create a new dataset, a new csv will always be created,
    with a trailing number if several copies of it already exists.

* Start :
    Once you are ready, you can press **Start reviewing** to start the review process.

.. warning::
    Starting a review session opens a new window and closes the current one.
    Make sure you have saved your work before starting a review session.
    This behaviour may change in the future.

Interface & functionalities
---------------------------------------------------------------

.. image:: ../images/review_process_example.png

Once you have launched the review process, you will have access to the following functionalities:

.. hlist::
   :columns: 1

   * A dialog to choose the folder in which you want to save the verified and/or corrected annotations, and a button to save the labels. They will be saved based on the file format you provided.
   * A button to update the status of the slice in the csv file (in this case : checked/not checked)
   * A plot with three projections in the x-y, y-z and x-z planes, to allow the reviewer to better see the surroundings of the label and properly establish whether the image should be labeled or not. You can **shift-click** anywhere on the image or label layer to update the plot to the location being reviewed.

Using these, you can check your labels, correct them, save them and keep track of which slices have been checked or not.

.. note::
    You can find the csv file containing the annotation status **in the same folder as the labels**

Source code
-------------------------------------------------

* :doc:`../code/plugin_review`
* :doc:`../code/plugin_review_dock`
* :doc:`../code/plugin_base`
