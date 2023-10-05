Welcome to CellSeg3D!
=============================================


**CellSeg3D is a toolbox for 3D segmentation of cells in light-sheet microscopy images, using napari**



.. figure:: ./source/images/plugin_welcome.png
    :align: center



This plugin will allow you to:

* Review labeled cell volumes from mice whole-brain samples imaged by mesoSPIM microscopy [1]_
* Train and use segmentation models from the MONAI project [2]_ or custom 3D segmentation models written in PyTorch.

Additionally, if you do not have labeled data, you can try our unsupervised model
to help you obtain labels for your data automatically.

The models provided should be adaptable to other tasks related to detection of 3D objects,
outside of whole-brain light-sheet microscopy.

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
    A **CUDA-capable GPU** is not needed but **very strongly recommended**, especially for training and possibly inference.

* For help with PyTorch, please see `PyTorch's website`_ for installation instructions, with or without CUDA according to your hardware.
  **Depending on your setup, you might wish to install torch first.**

* If you get errors from MONAI regarding missing readers, please see `MONAI's optional dependencies`_ page for instructions on getting the readers required by your images.

.. _MONAI's optional dependencies: https://docs.monai.io/en/stable/installation.html#installing-the-recommended-dependencies
.. _PyTorch's website: https://pytorch.org/get-started/locally/


Installation
--------------------------------------------

You can install `napari-cellseg3d` via pip:

.. code-block::

  pip install napari-cellseg3d

For local installation after cloning from GitHub, please run the following in the CellSeg3D folder:

.. code-block::

  pip install -e .

If the installation was successful, you will find the napari-cellseg3D plugin in the Plugins section of napari.

M1 Mac installation
________________________
To avoid issues when installing on the ARM64 architecture, please follow these steps.

1) Create a new conda env using the provided conda/napari_cellseg3d_m1.yml file :


.. code-block::

    git clone https://github.com/AdaptiveMotorControlLab/CellSeg3d.git
    cd CellSeg3d
    conda env create -f conda/napari_cellseg3d_m1.yml
    conda activate napari_cellseg3d_m1

2) Then install PyQt5 from conda separately :

.. code-block::

    conda install -c anaconda pyqt

3) And install the plugin :

.. code-block::

    pip install napari-cellseg3d


Optional requirements
________________________

In order to reduce install size, we provide some functionalities as optional requirements.
These are not installed by default and include :abbreviation:

* Optional modules:

  * CRF : Conditional Random Fields for post-processing of predictions from WNet, as suggested by Xia and Kulis [3_]

    .. code-block::

      pip install napari-cellseg3d[crf]

  * **WIP** WandB : WandB support for WNet training. This allows you to monitor your training on the WandB platform.
    See :ref:`WandB integration in Training <wandb_integration>` for more details.

    .. code-block::

      pip install napari-cellseg3d[wandb]
      wandb login

  * **WIP** ONNX model support, with or without GPU support. This allows you to run any ONNX model.


    To use this feature, select WNet during inference and load your ONNX in the custom weights field.
    This will run your ONNX model instead of the WNet.

    .. code-block::

      pip install napari-cellseg3d[onnx-cpu]

* Dev requirements (see *pyproject.toml* for details):

  * For local testing:

    .. code-block::

       pip install napari-cellseg3d[test]

  * For building the documentation locally:

    .. code-block::

       pip install napari-cellseg3d[docs]

  * Useful tools:

    .. code-block::

       pip install napari-cellseg3d[dev]

Usage
--------------------------------------------

To use the plugin, please run:

.. code-block::

    napari

Then go into **Plugins > napari-cellseg3d**, and choose the tool to use:

- **Review**: Review your labels, from predictions or manual labeling, and correct them if needed. It then saves the status of each file in a csv, for easier monitoring
- **Training**:  Train segmentation algorithms from labeled volumes
- **Inference**: Use pre-trained segmentation algorithms on volumes to automatically label cells
- **Utilities**: Use several utilities, e.g. to crop your volumes and labels, convert semantic labels to instance, and more
- **Help/About...** : Quick access to version info, Github pages and documentation

.. hint::
    Many buttons have tooltips to help you understand what they do.
    Simply Hover over them to see the tooltip.

See below for links to detailed guides regarding the usage of the modules.

Documentation contents
--------------------------------------------
From this page you can access the guides on the several modules available for your tasks, such as :

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

.. _napari-pyclesperanto-assistant: https://www.napari-hub.org/plugins/napari-pyclesperanto-assistant
.. _napari-brightness-contrast: https://www.napari-hub.org/plugins/napari-brightness-contrast
.. _brainreg-napari: https://www.napari-hub.org/plugins/brainreg-napari

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

* A custom re-implementation of the `WNet model`_ by Xia and Kulis [3]_

.. _Mathis Laboratory of Adaptive Motor Control: http://www.mackenziemathislab.org/
.. _Wyss Center: https://wysscenter.ch/
.. _TRAILMAP project on GitHub: https://github.com/AlbertPun/TRAILMAP
.. _napari: https://napari.org/
.. _PyTorch: https://pytorch.org/
.. _MONAI project: https://monai.io/
.. _on their website: https://docs.monai.io/en/stable/networks.html#nets
.. _pyclEsperanto: https://github.com/clEsperanto/pyclesperanto_prototype
.. _WNet model: https://arxiv.org/abs/1711.08506

.. rubric:: References

.. [1] The mesoSPIM initiative: open-source light-sheet microscopes for imaging cleared tissue, Voigt et al., 2019 ( https://doi.org/10.1038/s41592-019-0554-0 )
.. [2] MONAI Project website ( https://monai.io/ )
.. [3] W-Net: A Deep Model for Fully Unsupervised Image Segmentation, Xia and Kulis, 2018 ( https://arxiv.org/abs/1711.08506 )
