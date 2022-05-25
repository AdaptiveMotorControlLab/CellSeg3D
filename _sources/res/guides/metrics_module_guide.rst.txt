.. _metrics_module_guide:

Metrics utility guide
==========================

This utility allows to compute the Dice coefficient between two folders of labels.

The Dice coefficient is defined as :

.. math:: \frac {2|X \cap Y|} {|X|+|Y|}

It is a measure of similarity between two sets- :math:`0` indicating no similarity and :math:`1` complete similarity.

You will need to provide the following parameters:

* Two folders : one for ground truth labels and one for prediction labels.

* The threshold below which the score is considered insufficient.
  Any pair below that score will be shown on the viewer; and be displayed in red in the plot.

* Whether to automatically determine the best orientation for the computation by rotating and flipping;
  use this if your images do not have the same orientation.

.. note::
    Due to changes in orientation of images after running inference, the utility can rotate and flip images randomly to find the best Dice coefficient
    to compensate. If you have small images with a very large number of labels, this can lead to an inexact metric being computed.
    Images with a low score might be in the wrong orientation as well when displayed for comparison.

.. important::
    This utility assumes that **predictions are padded to a power of two already.** Ground truth labels can be smaller,
    they will be padded to match the prediction size.
    Your files should have names that allow them to be sorted numerically as well; please ensure that each ground truth label has a matching prediction label.

Once you are ready, press the "Compute Dice" button. This will plot the Dice score for each ground truth-prediction labels pair on the side.
Pairs with a low score will be displayed on the viewer for checking, ground truth in **blue**, low score prediction in **red**.




Source code
-------------------------------------------------

* :doc:`../code/plugin_base`
* :doc:`../code/plugin_metrics`




