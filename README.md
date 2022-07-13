# napari-cellseg3D: a napari plug-in for direct 3D cell segmentation with deep learning


<img src="https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/838605d0-9723-4e43-83cd-6dbfe4adf36b/cellseg-logo.png?format=1500w" title="cellseg3d" alt="cellseg3d logo" width="350" align="right" vspace = "80"/>

<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/AdaptiveMotorControlLab/CellSeg3d/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-cellseg3d.svg?color=green)](https://pypi.org/project/napari-cellseg3d)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-cellseg-annotator.svg?color=green)](https://python.org)
[![codecov](https://codecov.io/gh/AdaptiveMotorControlLab/CellSeg3d/branch/main/graph/badge.svg?token=hzUcn3XN8F)](https://codecov.io/gh/AdaptiveMotorControlLab/CellSeg3d)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-cellseg3d)](https://www.napari-hub.org/plugins/napari-cellseg3d)


A napari plugin for 3D cell segmentation: training, inference, and data review. In particular, this project was developed for analysis of mesoSPIM-acquired (cleared tissue + lightsheet) datasets.

----------------------------------

## News

**June 2022: This is an alpha version, please expect bugs and issues, and help us make the code better by reporting them as an issue!**



## Installation

Note : we recommend using conda to create a new environment for the plugin.

    conda create --name python=3.8 napari-cellseg3d
    conda activate napari-cellseg3d

You can install `napari-cellseg3d` via [pip]:  

    pip install napari-cellseg3d

OR directly via [napari-hub]:

- Install napari from pip with `pip install "napari[all]"`,
then from the “Plugins” menu within the napari application, select “Install/Uninstall Package(s)...”
- Copy `napari-cellseg3d` and paste it where it says “Install by name/url…”
- Click “Install”

## Documentation

Available at https://AdaptiveMotorControlLab.github.io/CellSeg3d

You can also generate docs by running ``make html`` in the docs folder.

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
**Python >= 3.8 required**

Requires **pytorch** and **MONAI**.
For PyTorch, please see [PyTorch's website for installation instructions].
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
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/

[PyTorch's website for installation instructions]: https://pytorch.org/get-started/locally/
[MONAI's optional dependencies]: https://docs.monai.io/en/stable/installation.html#installing-the-recommended-dependencies

## Acknowledgements

This plugin was developed by Cyril Achard, Maxime Vidal, Mackenzie Mathis. This work was funded, in part, from the Wyss Center to the [Mathis Laboratory of Adaptive Motor Control](https://www.mackenziemathislab.org/).


## Plugin base
This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/plugins/stable/index.html
-->
