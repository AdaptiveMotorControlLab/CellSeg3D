# CellSeg3D: self-supervised (and supervised) 3D cell segmentation
<img src="https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/838605d0-9723-4e43-83cd-6dbfe4adf36b/cellseg-logo.png?format=1500w" title="cellseg3d" alt="cellseg3d logo" width="350" align="right" vspace = "80"/>

<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
[![PyPI](https://img.shields.io/pypi/v/napari-cellseg3d.svg?color=green)](https://pypi.org/project/napari-cellseg3d)
[![Downloads](https://static.pepy.tech/badge/napari-cellseg3d)](https://pepy.tech/project/napari-cellseg3d)
[![Downloads](https://static.pepy.tech/badge/napari-cellseg3d/month)](https://pepy.tech/project/napari-cellseg3d)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/AdaptiveMotorControlLab/CellSeg3D/raw/main/LICENSE)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-cellseg-annotator.svg?color=green)](https://python.org)
[![codecov](https://codecov.io/gh/AdaptiveMotorControlLab/CellSeg3D/branch/main/graph/badge.svg?token=hzUcn3XN8F)](https://codecov.io/gh/AdaptiveMotorControlLab/CellSeg3D)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-cellseg3d)](https://www.napari-hub.org/plugins/napari-cellseg3d)

- A napari plugin for 3D cell segmentation: training, inference, and data review. In particular, this project was developed for analysis of mesoSPIM-acquired (cleared tissue + lightsheet) datasets.

![demo](https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/0d16a71b-3ff2-477a-9d83-18d96cb1ce28/full_demo.gif?format=500w)

## Installation

 💻 See the [Installation page] in the documentation for detailed instructions.

## Documentation

📚 A lot of documentation is available at https://AdaptiveMotorControlLab.github.io/CellSeg3D

You can also generate docs by running ``make html`` in the docs/ folder.

## Quick Start

To use the plugin, please run:
```
napari
```
Then go into Plugins > napari-cellseg3d, and choose which tool to use.

- **Review (label)**: This module allows you to review your labels, from predictions or manual labeling, and correct them if needed. It then saves the status of each file in a csv, for easier monitoring.
- **Inference**: This module allows you to use pre-trained segmentation algorithms on volumes to automatically label cells and compute statistics.
- **Train**:  This module allows you to train segmentation algorithms from labeled volumes.
- **Utilities**: This module allows you to perform several actions like cropping your volumes and labels dynamically, by selecting a fixed size volume and moving it around the image; fragment images into smaller cubes for training; or converting labels from instance to segmentation and the opposite.

## News

**New version : v0.2.0**

- Changed project name to "napari_cellseg3d" to avoid setuptools deprecation
- Small API changes for training/inference from a script
- Some fixes to WandB integration ad csv saving after training

Previous additions :

- v0.1.2 :Fixed manifest issue for PyPi
- Improved training interface
- Unsupervised model : WNet3D
  - Generate labels directly from raw data!
  - Can be trained in napari directly or in Google Colab
  - Pretrained weights for mesoSPIM whole-brain cell segmentation
- WandB support (install wandb and login to use automatically when training)
- Remade and improved documentation
  - Moved to Jupyter Book
  - Dedicated installation page, and working ARM64 install for macOS Silicon users
- New utilities
- Many small improvements and many bug fixes



### Install note for ARM64 (Silicon) Mac users

To avoid issues when installing on the ARM64 architecture, please follow these steps.

1) Create a new conda env using the provided conda/napari_CellSeg3D_ARM64.yml file :

        git clone https://github.com/AdaptiveMotorControlLab/CellSeg3d.git
        cd CellSeg3d
        conda env create -f conda/CellSeg3D_ARM64.yml
        conda activate napari_CellSeg3D_ARM64


2) Install a Qt backend (PySide or PyQt5)
3) Launch napari, the plugin should be available in the plugins menu.



## Requirements

**Python 3.8 or 3.9 required.**
Requires **[napari]**, **[PyTorch]** and **[MONAI]**.
Compatible with Windows, MacOS and Linux.
Installation should not take more than 30 minutes, depending on your internet connection.

For PyTorch, please see [the PyTorch website for installation instructions].

A CUDA-capable GPU is not needed but very strongly recommended, especially for training.

If you get errors from MONAI regarding missing readers, please see [MONAI's optional dependencies] page for instructions on getting the readers required by your images.

## Quick demo

After installation, you can run the plugin by running:

        napari

and launching the plugin from the Plugins menu.
You may use the test volume in the `examples` folder to test the inference and review tools.
This should run in far less than five minutes on a modern computer.

You may also find a demo Colab notebook in the `notebooks` folder.

## Issues

**Help us make the code better by reporting issues and adding your feature requests!**


If you encounter any problems, please [file an issue] along with a detailed description.

## Testing

Before testing, install all requirements using ``pip install napari-cellseg3d[test]``.

``pydensecrf`` is also required for testing.

To run tests locally:

- Locally : run ``pytest`` in the plugin folder
- Locally with coverage : In the plugin folder, run ``coverage run --source=napari_cellseg3d -m pytest`` then ``coverage xml`` to generate a .xml coverage file.
- With tox : run ``tox`` in the plugin folder (will simulate tests with several python and OS configs, requires substantial storage space)

## Contributing

Contributions are very welcome.

Please ensure the coverage at least stays the same before you submit a pull request.

For local installation from Github cloning, please run:

```
pip install -e .
```

## License

Distributed under the terms of the [MIT] license.

"napari-cellseg3d" is free and open source software.

[napari-hub]: https://www.napari-hub.org/plugins/napari-cellseg3d

[file an issue]: https://github.com/AdaptiveMotorControlLab/CellSeg3D/issues
[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
[Installation page]: https://adaptivemotorcontrollab.github.io/CellSeg3D/source/guides/installation_guide.html
[the PyTorch website for installation instructions]: https://pytorch.org/get-started/locally/
[PyTorch]: https://pytorch.org/get-started/locally/
[MONAI's optional dependencies]: https://docs.monai.io/en/stable/installation.html#installing-the-recommended-dependencies
[MONAI]: https://docs.monai.io/en/stable/installation.html#installing-the-recommended-dependencies

## Acknowledgements

This plugin was developed by originally Cyril Achard, Maxime Vidal, Mackenzie Mathis.
This work was funded, in part, from the Wyss Center to the [Mathis Laboratory of Adaptive Intelligence](https://www.mackenziemathislab.org/).
Please refer to the documentation for full acknowledgements.

## Plugin base

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.
