Introduction
===================

Welcome to napari-cellseg-3d !
--------------------------------------------

Here you will find instructions on how to use the program.
If the installation was successful, you'll see the napari-cellseg-3d plugin
in the Plugins section of napari.

This plugin is intended for the review of labeled cell volumes [#]_ from mice whole-brain samples
imaged by mesoSPIM microscopy [#]_ , and for training and using segmentation models from the MONAI project [#]_, or
any custom model written in Pytorch.

From here you can access the guides on the several modules available for your tasks, such as :

* Main modules :
    * Review : :ref:`loader_module_guide`
    * Inference: :ref:`inference_module_guide`
    * Training : :ref:`training_module_guide`
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

Requires manual installation of pytorch and some optional dependencies of MONAI.

* For Pytorch, please see `PyTorch's website`_ for installation instructions, with or without CUDA depending on your hardware.

* If you get errors from MONAI regarding missing readers, please see `MONAI's optional dependencies`_ page for instructions on getting the readers required by your images.

.. _MONAI's optional dependencies: https://docs.monai.io/en/stable/installation.html#installing-the-recommended-dependencies
.. _PyTorch's website: https://pytorch.org/get-started/locally/

Installation
--------------------------------------------

You can install `napari-cellseg-3d` via [pip]:

    ``pip install napari-cellseg-3d``

For local installation, please run:

    ``pip install -e .``



Usage
--------------------------------------------

To use the plugin, please run:

    ``napari``

Then go into Plugins > napari-cellseg-3d, and choose which tool to use.

- **Review**: This module allows you to review your labels, from predictions or manual labeling, and correct them if needed. It then saves the status of each file in a csv, for easier monitoring.
- **Infer**: This module allows you to use pre-trained segmentation algorithms on volumes to automatically label cells.
- **Train**:  This module allows you to train segmentation algorithms from labeled volumes.
- **Utilities**: This module allows you to use several utilities, e.g. to crop your volumes and labels, compute prediction scores or convert labels


Credits & acknowledgments
---------------------------------------------
This plugin has been developed by Cyril Achard and Maxime Vidal for the `Mathis Laboratory of Adaptive Motor Control`_.

The TRAILMAP models and original weights used here all originate from the `TRAILMAP project on GitHub`_

    **Mapping Mesoscale Axonal Projections in the Mouse Brain Using A 3D Convolutional Network**
    *Drew Friedmann, Albert Pun, Eliza L Adams, Jan H Lui, Justus M Kebschull, Sophie M Grutzner, Caitlin Castagnola, Marc Tessier-Lavigne, Liqun Luo*
    bioRxiv 812644; doi: https://doi.org/10.1101/812644

This plugin mainly uses the following libraries and software:

* `Napari website`_

* `Pytorch website`_

* `MONAI project website`_ (various models used here are credited `on their website`_)


.. _Mathis Laboratory of adaptive motor control: http://www.mackenziemathislab.org/
.. _TRAILMAP project on GitHub: https://github.com/AlbertPun/TRAILMAP
.. _Napari website: https://napari.org/
.. _Pytorch website: https://pytorch.org/
.. _MONAI project website: https://monai.io/
.. _on their website: https://docs.monai.io/en/stable/networks.html#nets


.. rubric:: References

.. [#] Mapping mesoscale axonal projections in the mouse brain using a 3D convolutional network, Friedmann et al., 2020 ( https://pnas.org/cgi/doi/10.1073/pnas.1918465117 )
.. [#] The mesoSPIM initiative: open-source light-sheet microscopes for imaging cleared tissue, Voigt et al., 2019 ( https://doi.org/10.1038/s41592-019-0554-0 )
.. [#] MONAI Project website ( https://monai.io/ )

