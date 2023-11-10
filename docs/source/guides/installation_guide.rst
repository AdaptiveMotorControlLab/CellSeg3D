Installation guide ⚙
======================
This guide outlines the steps for installing CellSeg3D and its dependencies. The plugin is compatible with Windows, Linux, and MacOS.

**Note for M1/M2 (ARM64) Mac Users:**
Please refer to the :ref:`section below <source/guides/installation_guide:M1/M2 (ARM64) Mac installation>` for specific instructions.

.. warning::
    If you encounter any issues during installation, feel free to open an issue on our `GitHub repository`_.

.. _GitHub repository: https://github.com/AdaptiveMotorControlLab/CellSeg3d/issues


Installing pre-requisites
---------------------------

PyQt5 or PySide2
_____________________

CellSeg3D requires either **PyQt5** or **PySide2** as a Qt backend for napari. If you don't have a Qt backend installed:

.. code-block::

    pip install napari[all]

This command installs PyQt5 by default.

PyTorch
_____________________

For PyTorch installation, refer to `PyTorch's website`_ , with or without CUDA according to your hardware.
Select the installation criteria that match your OS and hardware (GPU or CPU).

.. note::
    While a **CUDA-capable GPU** is not mandatory, it is highly recommended for both training and inference.


* Running into MONAI-related errors? Consult MONAI’s optional dependencies for solutions. Please see `MONAI's optional dependencies`_ page for instructions on getting the readers required by your images.

.. _MONAI's optional dependencies: https://docs.monai.io/en/stable/installation.html#installing-the-recommended-dependencies
.. _PyTorch's website: https://pytorch.org/get-started/locally/



Installing CellSeg3D
--------------------------------------------

.. warning::
    For M1 Mac users, please see the :ref:`section below <source/guides/installation_guide:M1/M2 (ARM64) Mac installation>`

**Via pip**:

.. code-block::

  pip install napari-cellseg3d

**Directly in napari**:

- Navigate to **Plugins > Install/Uninstall Packages**
- Search for ``napari-cellseg3d``

**For local installation** (after cloning from GitHub)
Navigate to the cloned CellSeg3D folder and run:

.. code-block::

  pip install -e .

Successful installation will add the napari-cellseg3D plugin to napari’s Plugins section.


M1/M2 (ARM64) Mac installation
-------------------------------
.. _ARM64_Mac_installation:

For ARM64 Macs, we recommend using our custom CONDA environment. This is particularly important for M1 or M2 MacBooks.

Start by installing `miniconda3`_.

.. _miniconda3: https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html

1. **Clone the repository** (`link <https://github.com/AdaptiveMotorControlLab/CellSeg3d>`_):

.. code-block::

    git clone https://github.com/AdaptiveMotorControlLab/CellSeg3d.git

2. **Create the Conda Environment** :
In the terminal, navigate to the CellSeg3D folder:

.. code-block::

    conda env create -f conda/napari_cellseg3d_m1.yml

3. **Activate the environment** :

.. code-block::

    conda activate napari_cellseg3d_m1

4. **Install PyQt5 via conda** :

.. code-block::

    conda install -c anaconda pyqt

5. **Install the plugin** :

.. code-block::

    pip install napari-cellseg3d
