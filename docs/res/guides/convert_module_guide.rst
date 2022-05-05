.. _convert_module_guide:

Label conversion utility
===========================

This utility will let you convert labels to various different formats.
You will have to specify the results directory for saving; afterwards you can run each action on a folder or on the currently selected layer.

You can :

* Convert to instance labels :
    This will convert 0/1 semantic labels to instance label, with a unique ID for each object using the watershed method.

* Convert to semantic labels :
    This will convert instance labels with unique IDs per object into 0/1 semantic labels, for example for training.

* Remove small objects :
    You can specify a size threshold in pixels; all objects smaller than this size will be removed in the image.