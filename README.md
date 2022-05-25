# napari-cellseg3d

[![License](https://img.shields.io/pypi/l/napari-cellseg-annotator.svg?color=green)](https://github.com/AdaptiveMotorControlLab/CellSeg3d/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-cellseg-annotator.svg?color=green)](https://pypi.org/project/napari-cellseg-annotator)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-cellseg-annotator.svg?color=green)](https://python.org)
[![tests](https://github.com/AdaptiveMotorControlLab/CellSeg3d/workflows/tests/badge.svg)](https://github.com/AdaptiveMotorControlLab/CellSeg3d/actions)
[![codecov](https://codecov.io/gh/AdaptiveMotorControlLab/CellSeg3d/branch/main/graph/badge.svg)](https://codecov.io/gh/AdaptiveMotorControlLab/CellSeg3d)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-cellseg-annotator)](https://napari-hub.org/plugins/cellseg-annotator-test)

napari plugin providing cell segmentation tools : training, inference, review...

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/plugins/stable/index.html
-->

## Requirements

**Python >= 3.8 required**

Requires manual installation of **pytorch** and **MONAI**.

For Pytorch, please see [PyTorch's website for installation instructions].
A CUDA-capable GPU is not needed but very strongly recommended, especially for training.

If you get errors from MONAI regarding missing readers, please see [MONAI's optional dependencies] page for instructions on getting the readers required by your images.


## Installation

You can install `napari-cellseg3d` via [pip]:

    pip install napari-cellseg3d

For local installation, please run:

```
pip install -e .
```

## Documentation

You can generate docs by running ``make html`` in the *docs* folder

## Usage

To use the plugin, please run:
```
napari
```
Then go into Plugins > napari-cellseg3d, and choose which tool to use.

- **Review**: This module allows you to review your labels, from predictions or manual labeling, and correct them if needed. It then saves the status of each file in a csv, for easier monitoring.
- **Infer**: This module allows you to use pre-trained segmentation algorithms on volumes to automatically label cells.
- **Train**:  This module allows you to train segmentation algorithms from labeled volumes.
- **Crop utility**: This module allows you to crop your volumes and labels dynamically, by selecting a fixed size volume and moving it around the image.

## Testing 

To run tests locally: 

- Locally : run ``pytest`` in the plugin folder
- Locally with coverage : In the plugin folder, run ``coverage run --source=napari_cellseg3d -m pytest`` then ``coverage.xml`` to generate a .xml coverage file.
- With tox : run ``tox`` in the plugin folder (will simulate tests with several python and OS configs, requires substantial storage space)

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [MIT] license,
"napari-cellseg3d" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

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