.. _training_module_guide:

Training module guide
=================================

This module allows you to train pre-defined Pytorch models for cell segmentation.
Pre-defined models are stored in napari-cellseg-annotator/models.

The training module is comprised of several tabs.
The first one will let you choose :

* The images folder
* The labels folder
* The model
* The number of samples to extract from each of your image to ensure correct size and perform data augmentation.

The second lets you define training parameters such as :

* The loss function used for training
* The batch size
* The number of epochs
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
perform data augmentation, select a CUDA device if one is present, and train the model.

.. note::
    You can stop the training at any time by clicking on the start button again.

    **The training will stop after the next validation interval is performed, to save the latest model should it be better.**

After two validations steps have been performed, the training loss values and validation metrics will be automatically plotted
and shown on napari every time a validation step completes.
