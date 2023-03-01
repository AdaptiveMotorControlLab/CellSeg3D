Introduction
===================


Here you will find instructions on how to use the plugin for direct segmentation in 3D.
If the installation was successful, you'll see the napari-cellseg3D plugin
in the Plugins section of napari.

This plugin was initially developed for the review of labeled cell volumes [#]_ from mice whole-brain samples
imaged by mesoSPIM microscopy [#]_ , and for training and using segmentation models from the MONAI project [#]_,
or any custom model written in PyTorch.
It should be adaptable to other tasks related to detection of 3D objects, as long as labels are available.


From this page you can access the guides on the several modules available for your tasks, such as :

* Main modules :
    * Training : :ref:`training_module_guide`
    * Inference: :ref:`inference_module_guide`
    * Review : :ref:`loader_module_guide`
* Utilities :
    * Cropping (3D) : :ref:`cropping_module_guide`
    * Other utilities : :ref:`utils_module_guide`

..
    * Convert labels : :ref:`utils_module_guide`
..
    * Compute scores : :ref:`metrics_module_guide`

* Advanced :
    * Defining custom models directly in the plugin (WIP) : :ref:`custom_model_guide`


Installation
--------------------------------------------

You can install `napari-cellseg3d` via [pip]:

  ``pip install napari-cellseg3d``

  For local installation, please run:

  ``pip install -e .``

Requirements
--------------------------------------------

.. important::
    A **CUDA-capable GPU** is not needed but **very strongly recommended**, especially for training.

This package requires you have napari installed first.

It also depends on PyTorch and some optional dependencies of MONAI. These come in the pip package above, but if
you need further assistance see below.

* For help with PyTorch, please see `PyTorch's website`_ for installation instructions, with or without CUDA depending on your hardware.

* If you get errors from MONAI regarding missing readers, please see `MONAI's optional dependencies`_ page for instructions on getting the readers required by your images.

.. _MONAI's optional dependencies: https://docs.monai.io/en/stable/installation.html#installing-the-recommended-dependencies
.. _PyTorch's website: https://pytorch.org/get-started/locally/


Usage
--------------------------------------------

To use the plugin, please run:

    ``napari``

Then go into Plugins > napari-cellseg3d, and choose which tool to use:


- **Review**: This module allows you to review your labels, from predictions or manual labeling, and correct them if needed. It then saves the status of each file in a csv, for easier monitoring
- **Inference**: This module allows you to use pre-trained segmentation algorithms on volumes to automatically label cells
- **Training**:  This module allows you to train segmentation algorithms from labeled volumes
- **Utilities**: This module allows you to use several utilities, e.g. to crop your volumes and labels, compute prediction scores or convert labels
- **Help/About...** : Quick access to version info, Github page and docs

See above for links to detailed guides regarding the usage of the modules.

Acknowledgments & References
---------------------------------------------
This plugin has been developed by Cyril Achard and Maxime Vidal, supervised by Mackenzie Mathis for the `Mathis Laboratory of Adaptive Motor Control`_.

We also greatly thank Timokleia Kousi for her contributions to this project and the `Wyss Center`_ for project funding.

The TRAILMAP models and original weights used here were ported to PyTorch but originate from the `TRAILMAP project on GitHub`_ [1]_.
We also provide a model that was trained in-house on mesoSPIM nuclei data in collaboration with Dr. Stephane Pages and Timokleia Kousi.

This plugin mainly uses the following libraries and software:

* `napari website`_

* `PyTorch website`_

* `MONAI project website`_ (various models used here are credited `on their website`_)


.. _Mathis Laboratory of Adaptive Motor Control: http://www.mackenziemathislab.org/
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
