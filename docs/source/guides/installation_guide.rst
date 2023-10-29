Installation guide ⚙
======================

In this guide we will detail how to install the plugin and its dependencies.
CellSeg3D is intended to run on Windows, Linux, or MacOS.

.. warning::
    If you encounter any issues during installation, please feel free to open an issue on our `repository`_.

M1/M2 (ARM64) Mac installation
-------------------------------

To avoid issues when installing on the ARM64 architecture, we recommend to use our supplied CONDA environment.
If you use M1 or M2 chip in your MacBook, it is recommended to install miniconda3, which operates on the same principles as anaconda.

Click for more information about `miniconda3`_.

.. _miniconda3: https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html

First, decide and navigate to the folder you wish to save the plugin (any folder will work).

Next, git clone our `repository`_ by running :

.. _repository: https://github.com/AdaptiveMotorControlLab/CellSeg3d

.. code-block::

    git clone https://github.com/AdaptiveMotorControlLab/CellSeg3d.git


Now, in terminal go to CellSeg3D folder and create a new conda environment:

.. code-block::

    conda env create -f conda/napari_cellseg3d_m1.yml

Now activate it:

.. code-block::

    conda activate napari_cellseg3d_m1

Then install PyQt5 from conda separately :

.. code-block::

    conda install -c anaconda pyqt

Lastly, install the plugin :

.. code-block::

    pip install napari-cellseg3d


Installing pre-requisites
---------------------------

PyQt5 or PySide2
_____________________

This package requires **PyQt5** or **PySide2** to be installed first for napari to run.
If you do not have a Qt backend installed you can use :

.. code-block::

    pip install napari[all]

to install PyQt5 by default.

PyTorch
_____________________

To install PyTorch, please see `PyTorch's website`_ for installation instructions, with or without CUDA according to your hardware.
Select the options relevant to your specific OS and hardware (GPU or CPU).

.. note::
    A **CUDA-capable GPU** is not needed but **very strongly recommended**, especially for training and to a lesser degree inference.

* If you get errors from MONAI regarding missing readers, please see `MONAI's optional dependencies`_ page for instructions on getting the readers required by your images.

.. _MONAI's optional dependencies: https://docs.monai.io/en/stable/installation.html#installing-the-recommended-dependencies
.. _PyTorch's website: https://pytorch.org/get-started/locally/



Installing the plugin
--------------------------------------------

.. warning::
    For M1 Mac users, please see the above :ref:`section <source/guides/installation_guide:M1/M2 (ARM64) Mac installation>`

You can install `napari-cellseg3d` via pip:

.. code-block::

  pip install napari-cellseg3d

or directly in napari by selecting **Plugins > Install/Uninstall Packages...** and searching for ``napari-cellseg3d``.

For local installation after cloning from GitHub, please run the following in the CellSeg3D folder:

.. code-block::

  pip install -e .

If the installation was successful, you will find the napari-cellseg3D plugin in the Plugins section of napari.
