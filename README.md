# napari-cellseg3D: a napari plug-in for direct 3D cell segmentation with deep learning

<img src="https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/838605d0-9723-4e43-83cd-6dbfe4adf36b/cellseg-logo.png?format=1500w" title="cellseg3d" alt="cellseg3d logo" width="350" align="right" vspace = "80"/>

<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/AdaptiveMotorControlLab/CellSeg3d/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-cellseg3d.svg?color=green)](https://pypi.org/project/napari-cellseg3d)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-cellseg-annotator.svg?color=green)](https://python.org)
[![codecov](https://codecov.io/gh/AdaptiveMotorControlLab/CellSeg3d/branch/main/graph/badge.svg?token=hzUcn3XN8F)](https://codecov.io/gh/AdaptiveMotorControlLab/CellSeg3d)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-cellseg3d)](https://www.napari-hub.org/plugins/napari-cellseg3d)

A napari plugin for 3D cell segmentation: training, inference, and data review. In particular, this project was developed for analysis of mesoSPIM-acquired (cleared tissue + lightsheet) datasets.

**Help us make the code better by reporting issues and adding your feature requests!**

----------------------------------

## News

**New version : v0.1.1**

Added :

- Improved training interface
- Unsupervised model : WNet
  - Generate labels directly from raw data !
  - Can be trained in napari directly or in Colab
  - Pretrained weights for mesoSPIM whole-brain cell segmentation
- WandB support (install wandb and login to use automatically when training)
- Remade and improved documentation
  - Moved to Jupyter Book
  - Dedicated installation page, and working ARM64 install for macOS Silicon users
- New utilities
- Many small improvements and many bug fixes

## Demo

![demo](https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/0d16a71b-3ff2-477a-9d83-18d96cb1ce28/full_demo.gif?format=500w)

## Installation

**Note** : we recommend using conda to create a new environment for the plugin.
**M1 Mac users, please see the [M1 install section](#m1-mac-users)**

    conda create --name napari-cellseg3d python=3.8
    conda activate napari-cellseg3d

You can install `napari-cellseg3d` via [pip]:

    pip install napari-cellseg3d[all]

OR directly via [napari-hub]:

- Install napari from pip with `pip install "napari[all]"`,
then from the “Plugins” menu within the napari application, select “Install/Uninstall Package(s)...”
- Copy `napari-cellseg3d` and paste it where it says “Install by name/url…”
- Click “Install”
- Restart napari

### M1 Mac users

To avoid issues when installing on the ARM64 architecture, please follow these steps.

1) Create a new conda env using the provided conda/napari_cellseg3d_m1.yml file :

        git clone https://github.com/AdaptiveMotorControlLab/CellSeg3d.git
        cd CellSeg3d
        conda env create -f conda/napari_cellseg3d_m1.yml
        conda activate napari_cellseg3d_m1

2) Install the plugin.
   From repository root folder, run :

        pip install -e .
   OR directly via PyPi :

        pip install napari-cellseg3d

   OR directly via [napari-hub] (see Installation section above)

## Documentation

Available at https://AdaptiveMotorControlLab.github.io/CellSeg3d

You can also generate docs by running ``make html`` in the docs/ folder.

## Usage

To use the plugin, please run:
```
napari
```
Then go into Plugins > napari-cellseg3d, and choose which tool to use.

- **Review**: This module allows you to review your labels, from predictions or manual labeling, and correct them if needed. It then saves the status of each file in a csv, for easier monitoring.
- **Inference**: This module allows you to use pre-trained segmentation algorithms on volumes to automatically label cells and compute statistics.
- **Train**:  This module allows you to train segmentation algorithms from labeled volumes.
- **Utilities**: This module allows you to perform several actions like cropping your volumes and labels dynamically, by selecting a fixed size volume and moving it around the image; computing prediction scores from ground truth and predicition labels; or converting labels from instance to segmentation and the opposite.

## Requirements

**Python 3.8 or 3.9 required.**
Requires **[napari]**, **[PyTorch]** and **[MONAI]**.

For PyTorch, please see [the PyTorch website for installation instructions].

A CUDA-capable GPU is not needed but very strongly recommended, especially for training.

If you get errors from MONAI regarding missing readers, please see [MONAI's optional dependencies] page for instructions on getting the readers required by your images.

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

## Testing

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

[file an issue]: https://github.com/AdaptiveMotorControlLab/CellSeg3d/issues
[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/

[the PyTorch website for installation instructions]: https://pytorch.org/get-started/locally/
[PyTorch]: https://pytorch.org/get-started/locally/
[MONAI's optional dependencies]: https://docs.monai.io/en/stable/installation.html#installing-the-recommended-dependencies
[MONAI]: https://docs.monai.io/en/stable/installation.html#installing-the-recommended-dependencies

## Acknowledgements

This plugin was developed by Cyril Achard, Maxime Vidal, Mackenzie Mathis.
This work was funded, in part, from the Wyss Center to the [Mathis Laboratory of Adaptive Motor Control](https://www.mackenziemathislab.org/).
Please refer to the documentation for full acknowledgements.

## Plugin base

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.
