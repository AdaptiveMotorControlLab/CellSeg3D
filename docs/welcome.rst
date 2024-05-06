Welcome to CellSeg3D!
=====================


.. figure:: ./source/images/plugin_welcome.png
    :align: center

**CellSeg3D** is a toolbox for 3D segmentation of cells in light-sheet microscopy images, using napari.
Use CellSeg3D to:

* Review labeled cell volumes from whole-brain samples of mice imaged by mesoSPIM microscopy [1]_
* Train and use segmentation models from the MONAI project [2]_ or implement your own custom 3D segmentation models using PyTorch.

No labeled data? Try our unsupervised model, based on the `WNet`_ model, to automate your data labelling.

The models provided should be adaptable to other tasks related to detection of 3D objects,
outside of whole-brain light-sheet microscopy.
This applies to the unsupervised model as well, feel free to try to generate labels for your own data!

.. figure:: https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/0d16a71b-3ff2-477a-9d83-18d96cb1ce28/full_demo.gif?format=500w
   :alt: CellSeg3D demo
   :width: 500
   :align: center

   Demo of the plugin


Requirements
--------------------------------------------

.. important::
    This package requires **PyQt5** or **PySide2** to be installed first for napari to run.
    If you do not have a Qt backend installed you can use :
    ``pip install napari[all]``
    to install PyQt5 by default.

This package depends on PyTorch and certain optional dependencies of MONAI. These come as requirements, but if
you need further assistance, please see below.

.. note::
    A **CUDA-capable GPU** is not needed but **very strongly recommended**, especially for training and to a lesser degree inference.

* For help with PyTorch, please see `PyTorch's website`_ for installation instructions, with or without CUDA according to your hardware.
  **Depending on your setup, you might wish to install torch first.**

* If you get errors from MONAI regarding missing readers, please see `MONAI's optional dependencies`_ page for instructions on getting the readers required by your images.

.. _MONAI's optional dependencies: https://docs.monai.io/en/stable/installation.html#installing-the-recommended-dependencies
.. _PyTorch's website: https://pytorch.org/get-started/locally/



Installation
--------------------------------------------
CellSeg3D can be run on Windows, Linux, or MacOS.

For detailed installation instructions, including installing pre-requisites,
please see :ref:`source/guides/installation_guide:Installation guide ⚙`

.. warning::
    **ARM64 MacOS users**, please refer to the :ref:`dedicated section <source/guides/installation_guide:ARM64 Mac installation>`

You can install ``napari-cellseg3d`` via pip:

.. code-block::

  pip install napari-cellseg3d

For local installation after cloning from GitHub, please run the following in the CellSeg3D folder:

.. code-block::

  pip install -e .

If the installation was successful, you will find the napari-cellseg3D plugin in the Plugins section of napari.


Usage
--------------------------------------------


To use the plugin, please run:

.. code-block::

    napari

Then go into **Plugins > CellSeg3D**

.. figure:: ./source/images/plugin_menu.png
    :align: center


and choose the correct tool to use:

- :ref:`review_module_guide`: Examine and refine your labels, whether manually annotated or predicted by a pre-trained model.
- :ref:`training_module_guide`:  Train segmentation algorithms on your own data.
- :ref:`inference_module_guide`: Use pre-trained segmentation algorithms on volumes to automate cell labelling.
- :ref:`utils_module_guide`: Leverage various utilities, including cropping your volumes and labels, converting semantic to instance labels, and more.
- **Help/About...** : Quick access to version info, Github pages and documentation.

.. hint::
    Many buttons have tooltips to help you understand what they do.
    Simply hover over them to see the tooltip.


Documentation contents
--------------------------------------------
_`From this page you can access the guides on the several modules available for your tasks`, such as :


* Main modules :
    * :ref:`review_module_guide`
    * :ref:`training_module_guide`
    * :ref:`inference_module_guide`
* Utilities :
    * :ref:`cropping_module_guide`
    * :ref:`utils_module_guide`

..
    * Convert labels : :ref:`utils_module_guide`
..
    * Compute scores : :ref:`metrics_module_guide`

* Advanced :
    * :ref:`training_wnet`
    * :ref:`custom_model_guide` **(WIP)**

Other useful napari plugins
---------------------------------------------

.. important::
    | Please note that these plugins are not developed by us, and we cannot guarantee their compatibility, functionality or support.
    | Installing napari plugins in separated environments is recommended.

* `brainreg-napari`_ : Whole-brain registration in napari
* `napari-brightness-contrast`_ : Adjust brightness and contrast of your images, visualize histograms and more
* `napari-pyclesperanto-assistant`_ : Image processing workflows using pyclEsperanto
* `napari-skimage-regionprops`_ : Compute region properties on your labels

.. _napari-pyclesperanto-assistant: https://www.napari-hub.org/plugins/napari-pyclesperanto-assistant
.. _napari-brightness-contrast: https://www.napari-hub.org/plugins/napari-brightness-contrast
.. _brainreg-napari: https://www.napari-hub.org/plugins/brainreg-napari
.. _napari-skimage-regionprops: https://www.napari-hub.org/plugins/napari-skimage-regionprops

Acknowledgments & References
---------------------------------------------
This plugin has been developed by Cyril Achard and Maxime Vidal, supervised by Mackenzie Mathis for the `Mathis Laboratory of Adaptive Motor Control`_.

We also greatly thank Timokleia Kousi for her contributions to this project and the `Wyss Center`_ for project funding.

The TRAILMAP models and original weights used here were ported to PyTorch but originate from the `TRAILMAP project on GitHub`_.
We also provide a model that was trained in-house on mesoSPIM nuclei data in collaboration with Dr. Stephane Pages and Timokleia Kousi.

This plugin mainly uses the following libraries and software:

* `napari`_

* `PyTorch`_

* `MONAI project`_ (various models used here are credited `on their website`_)

* `pyclEsperanto`_ (for the Voronoi Otsu labeling) by Robert Haase

* A new unsupervised 3D model based on the `WNet`_ by Xia and Kulis [3]_

.. _Mathis Laboratory of Adaptive Motor Control: http://www.mackenziemathislab.org/
.. _Wyss Center: https://wysscenter.ch/
.. _TRAILMAP project on GitHub: https://github.com/AlbertPun/TRAILMAP
.. _napari: https://napari.org/
.. _PyTorch: https://pytorch.org/
.. _MONAI project: https://monai.io/
.. _on their website: https://docs.monai.io/en/stable/networks.html#nets
.. _pyclEsperanto: https://github.com/clEsperanto/pyclesperanto_prototype
.. _WNet: https://arxiv.org/abs/1711.08506

.. rubric:: References

.. [1] The mesoSPIM initiative: open-source light-sheet microscopes for imaging cleared tissue, Voigt et al., 2019 ( https://doi.org/10.1038/s41592-019-0554-0 )
.. [2] MONAI Project website ( https://monai.io/ )
.. [3] W-Net: A Deep Model for Fully Unsupervised Image Segmentation, Xia and Kulis, 2018 ( https://arxiv.org/abs/1711.08506 )
