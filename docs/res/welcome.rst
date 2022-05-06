Introduction
===================

Welcome to napari-cellseg3D !
--------------------------------------------

Here you will find instructions on how to use the plug-in program.
If the installation was successful, you'll see the napari-cellseg3D plugin
in the Plugins section of napari.

This plugin is intended for the review of labeled cell volumes [#]_ from mice whole-brain samples
imaged by mesoSPIM microscopy [#]_ , and for training and using segmentation models from the MONAI project [#]_, or
any custom model written in Pytorch.

From here you can access the guides on the several modules available for your tasks, such as :

* Main modules :
    * Training : :ref:`training_module_guide`
    * Inference: :ref:`inference_module_guide`
    * Review : :ref:`loader_module_guide`
* Utilities :
    * Cropping (3D) : :ref:`cropping_module_guide`
    * Convert labels : :ref:`convert_module_guide`
    * Compute scores : :ref:`metrics_module_guide`
* Advanced :
    * Defining custom models directly in the plugin (WIP) : :ref:`custom_model_guide`


Requirements
--------------------------------------------

.. important::
    A **CUDA-capable GPU** is not needed but **very strongly recommended**, especially for training.

Requires installation of PyTorch and some optional dependencies of MONAI.

* For PyTorch, please see `PyTorch's website`_ for installation instructions, with or without CUDA depending on your hardware.

* If you get errors from MONAI regarding missing readers, please see `MONAI's optional dependencies`_ page for instructions on getting the readers required by your images.

.. _MONAI's optional dependencies: https://docs.monai.io/en/stable/installation.html#installing-the-recommended-dependencies
.. _PyTorch's website: https://pytorch.org/get-started/locally/

Installation
--------------------------------------------

You can install `napari-cellseg3D` via [pip]:

    ``pip install napari-cellseg3D``

For local installation, please run:

    ``pip install -e .``



Usage
--------------------------------------------

To use the plugin, please run:

    ``napari``

Then go into Plugins > napari-cellseg3D, and choose which tool to use.

- **Train**:  This module allows you to train segmentation algorithms from labeled volumes.
- **Infer**: This module allows you to use pre-trained segmentation algorithms on volumes to automatically label cells.
- **Review**: This module allows you to review your labels, from predictions or manual labeling, and correct them if needed. It then saves the status of each file in a csv, for easier monitoring.
- **Utilities**: This module allows you to use several utilities, e.g. to crop your volumes and labels, compute prediction scores or convert labels


Acknowledgments & References
---------------------------------------------
This plugin has been developed by Cyril Achard and Maxime Vidal for the `Mathis Laboratory of Adaptive Motor Control`_. We also greatly Timokleia Kousi for her contributions to this project and the `Wyss Center` for project funding.

The TRAILMAP models and original weights used here all originate from the `TRAILMAP project on GitHub`_ [1]_.

This plugin mainly uses the following libraries and software:

* `napari website`_

* `PyTorch website`_

* `MONAI project website`_ (various models used here are credited `on their website`_)


.. _Mathis Laboratory of adaptive motor control: http://www.mackenziemathislab.org/
.. _Wyss Center: https://wysscenter.ch/
.. _TRAILMAP project on GitHub: https://github.com/AlbertPun/TRAILMAP
.. _napari website: https://napari.org/
.. _PyTorch website: https://pytorch.org/
.. _MONAI project website: https://monai.io/
.. _on their website: https://docs.monai.io/en/stable/networks.html#nets


.. rubric:: References

.. [#] Mapping mesoscale axonal projections in the mouse brain using a 3D convolutional network, Friedmann et al., 2020 ( https://pnas.org/cgi/doi/10.1073/pnas.1918465117 )
.. [#] The mesoSPIM initiative: open-source light-sheet microscopes for imaging cleared tissue, Voigt et al., 2019 ( https://doi.org/10.1038/s41592-019-0554-0 )
.. [#] MONAI Project website ( https://monai.io/ )

