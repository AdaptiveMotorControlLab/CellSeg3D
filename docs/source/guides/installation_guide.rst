Installation guide ⚙
======================
This guide outlines the steps for installing CellSeg3D and its dependencies. The plugin is compatible with Windows, Linux, and MacOS.

**Note for ARM64 Mac Users:**
Please refer to the :ref:`section below <source/guides/installation_guide:ARM64 Mac installation>` for specific instructions.

.. warning::
    If you encounter any issues during installation, feel free to open an issue on our `GitHub repository`_.

.. _GitHub repository: https://github.com/AdaptiveMotorControlLab/CellSeg3D/issues


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
    For ARM64 Mac users, please see the :ref:`section below <source/guides/installation_guide:ARM64 Mac installation>`

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


ARM64 Mac installation
--------------------------------------------
.. _ARM64_Mac_installation:

For ARM64 Macs, we recommend using our custom CONDA environment. This is particularly important for ARM64 (Silicon chips) MacBooks.

Start by installing `miniconda3`_.

Creating the environment
______________________________

.. _miniconda3: https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html

1. **Clone the repository** (`link <https://github.com/AdaptiveMotorControlLab/CellSeg3D>`_):

.. code-block::

    git clone https://github.com/AdaptiveMotorControlLab/CellSeg3D.git

2. **Create the Conda Environment** :
In the terminal, navigate to the CellSeg3D folder:

.. code-block::

    cd CellSeg3D
    conda env create -f conda/napari_cellseg3d_ARM64.yml

This will also install the necessary dependencies as well as the plugin.

3. **Activate the environment** :

.. code-block::

    conda activate napari_cellseg3d_ARM64

4. **Install a Qt backend** :
Important : you only need to install one of the following backends.
PyQt5:

.. code-block::

        pip install PyQt5

OR
PySide2:

.. code-block::

        pip install PySide2

5. **Install PyTorch** :
Refer to `PyTorch's website`_ for installation instructions.

6. **Launch napari** :
You should now see the CellSeg3D plugin in the Plugins section of napari.
See `Usage section <https://adaptivemotorcontrollab.github.io/CellSeg3d/welcome.html#usage>`_ for a guide on how to use the plugin.

Updating the environment
______________________________

In order to update the environment, navigate to the CellSeg3D folder and run:

.. code-block::

    conda deactivate
    conda env update -f conda/napari_cellseg3d_ARM64.yml

Optional requirements
------------------------------

Additional functionalities
______________________________

Several additional functionalities are available optionally. To install them, use the following commands:

- CRF post-processing:

.. code-block::

    pip install pydensecrf@git+https://github.com/lucasb-eyer/pydensecrf.git#egg=master

- Weights & Biases integration:

.. code-block::

    pip install napari-cellseg3D[wandb]


- ONNX model support (EXPERIMENTAL):
  Depending on your hardware, you can install the CPU or GPU version of ONNX.

.. code-block::

    pip install napari-cellseg3D[onnx-cpu]
    pip install napari-cellseg3D[onnx-gpu]

Development requirements
______________________________

- Building the documentation:

.. code-block::

    pip install napari-cellseg3D[docs]

- Running tests locally:

.. code-block::

    pip install pydensecrf@git+https://github.com/lucasb-eyer/pydensecrf.git#egg=master
    pip install napari-cellseg3D[test]

- Dev utilities:

.. code-block::

    pip install napari-cellseg3D[dev]
