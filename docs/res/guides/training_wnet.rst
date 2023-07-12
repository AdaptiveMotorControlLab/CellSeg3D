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
The SoftNcuts is bounded between 0 and 1; the MSE may take large values.

For good performance, one should wait for the SoftNCut to reach a plateau, the reconstruction loss must also diminish but it's generally less critical.


Common issues troubleshooting
------------------------------
If you do not find a satisfactory answer here, please `open an issue`_ !

- **The NCuts loss explodes after a few epochs** : Lower the learning rate

- **The NCuts loss does not converge and is unstable** :
  The normalization step might not be adapted to your images. Disable normalization and change intensity_sigma according to the distribution of values in your image; for reference, by default images are remapped to values between 0 and 100, and intensity_sigma=1.

- **Reconstruction (decoder) performance is poor** : switch to BCE and set the scaling factor of the reconstruction loss ot 0.5, OR adjust the weight of the MSE loss to make it closer to 1.


.. _WNet, A Deep Model for Fully Unsupervised Image Segmentation: https://arxiv.org/abs/1711.08506
.. _open an issue: https://github.com/AdaptiveMotorControlLab/CellSeg3d/issues
