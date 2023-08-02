.. _training_wnet:

WNet model training
===================

This plugin provides a reimplemented, custom version of the WNet model from `WNet, A Deep Model for Fully Unsupervised Image Segmentation`_.
In order to train your own model, you may use the provided Jupyter notebook; support for in-plugin training might be added in the future.

The WNet uses brightness to cluster objects vs background; to get the most out of the model please use image regions with minimal
artifacts. You may then train one of the supervised models in order to achieve more resilient segmentation if you have many artifacts.

The WNet should not require a very large amount of data to train, but during inference images should be similar to those
the model was trained on; you can retrain from our pretrained model to your set of images to quickly reach good performance.

The model has two losses, the SoftNCut loss which clusters pixels according to brightness, and a reconstruction loss, either
Mean Square Error (MSE) or Binary Cross Entropy (BCE).
Unlike the original paper, these losses are added in a weighted sum and the backward pass is performed for the whole model at once.
The SoftNcuts is bounded between 0 and 1; the MSE may take large positive values.

For good performance, one should wait for the SoftNCut to reach a plateau; the reconstruction loss must also diminish but it's generally less critical.

Parameters
-------------------------------

When using the WNet training module, additional options will be provided in the Advanced tab of the training module:

- Number of classes : number of classes to segment (default 2). Additional classes will result in a more progressive segmentation according to brightness; can be useful if you have "halos" around your objects or artifacts with a significantly different brightness.
- Reconstruction loss : either MSE or BCE (default MSE). MSE is more sensitive to outliers, but can be more precise; BCE is more robust to outliers but can be less precise.

- NCuts parameters:
    - Intensity sigma : standard deviation of the feature similarity term (brightness here, default 1)
    - Spatial sigma : standard deviation of the spatial proximity term (default 4)
    - Radius : radius of the loss computation in pixels (default 2)

.. note::
    Intensity sigma depends on pixel values in the image. The default of 1 is tailored to images being mapped between 0 and 100, which is done automatically by the plugin.
.. note::
    Raising the radius might improve performance in some cases, but will also greatly increase computation time.

- Weights for the sum of losses :
    - NCuts weight : weight of the NCuts loss (default 0.5)
    - Reconstruction weight : weight of the reconstruction loss (default 0.5*1e-2)

.. note::
    The weight of the reconstruction loss should be adjusted according to its empirical value; ideally the reconstruction loss should be of the same order of magnitude as the NCuts loss after being multiplied by its weight.

Common issues troubleshooting
------------------------------
If you do not find a satisfactory answer here, please do not hesitate to `open an issue`_ on GitHub.

- **The NCuts loss explodes after a few epochs** : Lower the learning rate, first by a factor of two, then ten.

- **The NCuts loss does not converge and is unstable** :
  The normalization step might not be adapted to your images. Disable normalization and change intensity_sigma according to the distribution of values in your image. For reference, by default images are remapped to values between 0 and 100, and intensity_sigma=1.

- **Reconstruction (decoder) performance is poor** : switch to BCE and set the scaling factor of the reconstruction loss ot 0.5, OR adjust the weight of the MSE loss to make it closer to 1 in the weighted sum.


.. _WNet, A Deep Model for Fully Unsupervised Image Segmentation: https://arxiv.org/abs/1711.08506
.. _open an issue: https://github.com/AdaptiveMotorControlLab/CellSeg3d/issues
