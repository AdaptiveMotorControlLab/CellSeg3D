.. _training_module_guide:

Training module guide
=================================

This module allows you to train pre-defined Pytorch models for cell segmentation.
Pre-defined models are stored in napari-cellseg-annotator/models.

.. important::
    The machine learning models used by this program require all images of a dataset to all be of the same size.
    Please ensure that all the images you are loading are of the **same size**, or to use the **"extract patches" (in augmentation tab)** with an appropriately small size
    to ensure all images being used are of a proper size.

The training module is comprised of several tabs.


1) The first one, **Data**, will let you choose :

* The images folder
* The labels folder

2) The second tab, **Augmentation**, lets you define dataset and augmentation parameters such as :

* Whether to use images "as is" (**requires all images to be of the same size and cubic**) or extract patches.

* If you're extracting patches :
    * The size of patches to be extracted (ideally, please use a value **close to a pwoer of two**, such as 120 or 60.
    * The number of samples to extract from each of your image to ensure correct size and perform data augmentation. A larger number will likely mean better performances, but longer training and larger memory usage.
* Whether to perform data augmentation or not (elastic deforms, intensity shifts. random flipping,etc). A rule of thumb for augmentation is :
    * If you're using the patch extraction method, enable it if you have more than 10 samples per image with at least 5 images
    * If you have a large dataset and are not using patches extraction, enable it.


3) The third contains training related parameters :
* The model to use for training
* The loss function used for training (see table below)
* The batch size (larger means quicker training and possibly better performance but increased memory usage)
* The number of epochs (a possibility is to start with 60 epochs, and decrease or increase depending on performance.)
* The epoch interval for validation (for example, if set to two, the module will use the validation dataset to evaluate the model with the dice metric every two epochs.)
If the dice metric is better on that validation interval, the model weights will be saved in the results folder.

The available loss functions are :

========================  ====================================================
Function                  Reference
========================  ====================================================
Dice loss                 `Dice Loss from MONAI`_ with ``sigmoid=true``
Focal loss                `Focal Loss from MONAI`_
Dice-Focal loss           `Dice-focal Loss from MONAI`_ with ``sigmoid=true`` and ``lambda_dice = 0.5``
Generalized Dice loss     `Generalized dice Loss from MONAI`_ with ``sigmoid=true``
Dice-CE loss              `Dice-CE Loss from MONAI`_ with ``sigmoid=true``
Tversky loss              `Tversky Loss from MONAI`_ with ``sigmoid=true``
========================  ====================================================

.. _Dice Loss from MONAI: https://docs.monai.io/en/stable/losses.html#diceloss
.. _Focal Loss from MONAI: https://docs.monai.io/en/stable/losses.html#focalloss
.. _Dice-focal Loss from MONAI: https://docs.monai.io/en/stable/losses.html#dicefocalloss
.. _Generalized dice Loss from MONAI: https://docs.monai.io/en/stable/losses.html#generalizeddiceloss
.. _Dice-CE Loss from MONAI: https://docs.monai.io/en/stable/losses.html#diceceloss
.. _Tversky Loss from MONAI: https://docs.monai.io/en/stable/losses.html#tverskyloss

Once you are ready, press the Start button to begin training. The module will automatically load your dataset,
perform data augmentation if you chose to, select a CUDA device if one is present, and train the model.

.. note::
    You can stop the training at any time by clicking on the start button again.

    **The training will stop after the next validation interval is performed, to save the latest model should it be better.**

.. note::
    You can save the log to record the losses and validation metrics numerical value at each step. This log is autosaved as well when training completes.

After two validations steps have been performed, the training loss values and validation metrics will be automatically plotted
and shown on napari every time a validation step completes.
This plot automatically saved each time validation is performed for now. The final version is stored separately in the results folder.


