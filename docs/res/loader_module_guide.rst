.. _loader_module_guide:

Loader module guide
====================

This module allows you to review your labels, from predictions or manual labeling,
and correct them if needed. It then saves the status of each file in a csv, for easier monitoring.



Launching the review process
---------------------------------

First, you will be asked to provide a volume folder and a label folder, as well as the file extension
of your images (either .png or .tif).

.. note::
    Depending on the filetype you selected, the folders should either contain:

    * For .png, one png per slice (provide a folder of several pngs)
    * For .tif, a folder containing a single 3D tif file with all slices (if there are several, the first one will be used)

You can then provide a model name, which will be used in the csv file recording the status of each slice.

If a corresponding csv file exists already, it will be used. If not, a new one will be created.

If you choose to create a new dataset, a new csv will be created no matter what,
with a trailing number if several copies of it already exists.

Once you are ready, you can press **Run** to start the review process.

Review process : interface & functionalities
---------------------------------------------------------------

