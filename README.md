# CellSeg3D: self-supervised (and supervised) 3D cell segmentation
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-cellseg3d)](https://www.napari-hub.org/plugins/napari-cellseg3d)
[![PyPI](https://img.shields.io/pypi/v/napari-cellseg3d.svg?color=green)](https://pypi.org/project/napari-cellseg3d)
[![Downloads](https://static.pepy.tech/badge/napari-cellseg3d)](https://pepy.tech/project/napari-cellseg3d)
[![Downloads](https://static.pepy.tech/badge/napari-cellseg3d/month)](https://pepy.tech/project/napari-cellseg3d)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/AdaptiveMotorControlLab/CellSeg3D/raw/main/LICENSE)
[![codecov](https://codecov.io/gh/AdaptiveMotorControlLab/CellSeg3D/branch/main/graph/badge.svg?token=hzUcn3XN8F)](https://codecov.io/gh/AdaptiveMotorControlLab/CellSeg3D)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

<img src="https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/838605d0-9723-4e43-83cd-6dbfe4adf36b/cellseg-logo.png?format=1500w" title="cellseg3d" alt="cellseg3d logo" width="350" align="right" vspace = "80"/>


**A package for 3D cell segmentation with deep learning, including a napari plugin**: training, inference, and data review. In particular, this project was developed for analysis of mesoSPIM-acquired (cleared tissue + lightsheet) brain tissue datasets, but is not limited to this type of data. [Check out our preprint for more information!](https://www.biorxiv.org/content/10.1101/2024.05.17.594691v1)


![demo](https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/0d16a71b-3ff2-477a-9d83-18d96cb1ce28/full_demo.gif?format=500w)


## Installation

 💻 See the [Installation page](https://adaptivemotorcontrollab.github.io/CellSeg3D/welcome.html) in the documentation for detailed instructions.

## Documentation

📚 Documentation is available at [https://AdaptiveMotorControlLab.github.io/CellSeg3D
](https://adaptivemotorcontrollab.github.io/CellSeg3D/welcome.html)
You can also generate docs locally by running ``make html`` in the docs/ folder.

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

## Why use CellSeg3D?

The strength of our approach is we can match supervised model performance with purely self-supervised learning, meaning users don't need to spend (hundreds) of hours on annotation. Here is a quick look of our key results. TL;DR see panel **f**, which shows that with minmal input data we can outperform supervised models:

<p align="center">
<img src="https://www.biorxiv.org/content/biorxiv/early/2024/05/17/2024.05.17.594691/F1.large.jpg?format=200w" alt="Figure1" width="600"/>
</p>

#### Performance of 3D Semantic and Instance Segmentation Models.
**a:** Raw mesoSPIM whole-brain sample, volumes and corresponding ground truth labels from somatosensory (S1) and visual (V1) cortical regions.
**b:** Evaluation of instance segmentation performance for several supervised models over three data subsets. F1-score is computed from the Intersection over Union (IoU) with ground truth labels, then averaged. Error bars represent 50% Confidence Intervals (CIs).
**c:** View of 3D instance labels from supervised models, as noted, for visual cortex volume in b evaluation.
**d:** Illustration of our WNet3D architecture showcasing the dual 3D U-Net structure with modifications (see Methods).
**e:** Example 3D instance labels from WNet3D; top row is S1, bottom is V1, with artifacts removed.
**f:** Semantic segmentation performance: comparison of model efficiency, indicating the volume of training data required to achieve a given performance level. Each supervised model was trained with an increasing percentage of training data (with 10, 20, 60 or 80%, left to right within each model grouping); Dice score was computed on unseen test data, over three data subsets for each training/evaluation split. Our self-supervised model (WNet3D) is also trained on a subset of the training set of images, but always without human labels. Far right: We also show performance of the pretrained WNet3D available in the plugin (far right), with and without removing artifacts in the image. See Methods for details. The central box represents the interquartile range (IQR) of values with the median as a horizontal line, the upper and lower limits the upper and lower quartiles. Whiskers extend to data points within 1.5 IQR of the quartiles.
**g:** Instance segmentation performance comparison of Swin-UNetR and WNet3D (pretrained, see Methods), evaluated on unseen data across 3 data subsets, compared with a Swin-UNetR model trained using labels from the WNet3D self-supervised model. Here, WNet3D was trained on separate data, producing semantic labels that were then used to train a supervised Swin-UNetR model, still on held-out data. This supervised model was evaluated as the other models, on 3 held-out images from our dataset, unseen during training. Error bars indicate 50% CIs.


## News

**New version: v0.2.0**

- Changed project name to "napari_cellseg3d" to avoid setuptools deprecation
- Small API changes for training/inference from a script
- Some fixes to WandB integration ad csv saving after training

Previous additions:

- v0.1.2: Fixed manifest issue for PyPi
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




## Requirements

**Compatible with Python 3.8 to 3.10.**
Requires **[napari]**, **[PyTorch]** and **[MONAI]**.
Compatible with Windows, MacOS and Linux.
Installation should not take more than 30 minutes, depending on your internet connection.

For PyTorch, please see [the PyTorch website for installation instructions].

A CUDA-capable GPU is not needed but very strongly recommended, especially for training.

If you get errors from MONAI regarding missing readers, please see [MONAI's optional dependencies] page for instructions on getting the readers required by your images.

### Install note for ARM64 (Silicon) Mac users

To avoid issues when installing on the ARM64 architecture, please follow these steps.

1) Create a new conda env using the provided conda/napari_CellSeg3D_ARM64.yml file :

        git clone https://github.com/AdaptiveMotorControlLab/CellSeg3d.git
        cd CellSeg3d
        conda env create -f conda/CellSeg3D_ARM64.yml
        conda activate napari_CellSeg3D_ARM64


2) Install a Qt backend (PySide or PyQt5)
3) Launch napari, the plugin should be available in the plugins menu.



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

## Citation

```
@article {Achard2024,
	author = {Achard, Cyril and Kousi, Timokleia and Frey, Markus and Vidal, Maxime and Paychere, Yves and Hofmann, Colin and Iqbal, Asim and Hausmann, Sebastien B. and Pages, Stephane and Mathis, Mackenzie W.},
	title = {CellSeg3D: self-supervised 3D cell segmentation for microscopy},
	elocation-id = {2024.05.17.594691},
	year = {2024},
	doi = {10.1101/2024.05.17.594691},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2024/05/17/2024.05.17.594691},
	eprint = {https://www.biorxiv.org/content/early/2024/05/17/2024.05.17.594691.full.pdf},
	journal = {bioRxiv}
}
```
## Acknowledgements

This plugin was developed by originally Cyril Achard, Maxime Vidal, Mackenzie Mathis.
This work was funded, in part, from the Wyss Center to the [Mathis Laboratory of Adaptive Intelligence](https://www.mackenziemathislab.org/).
Please refer to the documentation for full acknowledgements.

## Plugin base

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.
