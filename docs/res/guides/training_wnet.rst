.. _training_wnet:

WNet model training
===================

This plugin provides a reimplemented version of the WNet model from `WNet, A Deep Model for Fully Unsupervised Image Segmentation`_.
In order to train your own model, you may use the provided Jupyter notebook.

The WNet uses brightness to cluster objects vs background; to get the most out of the model please use image regions with minimal
artifacts. You may then use one of the supervised models to train in order to achieve more resilient segmentation if you have many artifacts.

The WNet should not require a very large amount of data to train, but during inference images should be similar to those
the model was trained on.


.. _WNet, A Deep Model for Fully Unsupervised Image Segmentation: https://arxiv.org/abs/1711.08506
