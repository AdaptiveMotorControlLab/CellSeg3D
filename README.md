# napari-cellseg-annotator

[![License](https://img.shields.io/pypi/l/napari-cellseg-annotator.svg?color=green)](https://github.com/C_Achard/napari-cellseg-annotator/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-cellseg-annotator.svg?color=green)](https://pypi.org/project/napari-cellseg-annotator)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-cellseg-annotator.svg?color=green)](https://python.org)
[![tests](https://github.com/C_Achard/napari-cellseg-annotator/workflows/tests/badge.svg)](https://github.com/C_Achard/napari-cellseg-annotator/actions)
[![codecov](https://codecov.io/gh/C_Achard/napari-cellseg-annotator/branch/main/graph/badge.svg)](https://codecov.io/gh/C_Achard/napari-cellseg-annotator)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-cellseg-annotator)](https://napari-hub.org/plugins/napari-cellseg-annotator)

annotator for cell segmentation

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/plugins/stable/index.html
-->

## Installation

You can install `napari-cellseg-annotator` via [pip]:

    pip install napari-cellseg-annotator

For local installation, please run:

```
pip install -e .
```


## Usage

To use the plugin, please run:
```
napari
```
Then go into Plugins > napari-cellseg-annotator, and choose which tool to use.

- **Reviewer**: This module allows you to review your labels, from predictions or manual labeling, and correct them if needed. It then saves the status of each file in a csv, for easier monitoring.
- **Inferer**: This module allows you to use pre-trained segmentation algorithms on volumes to automatically label cells.
- **Trainer**:  This module allows you to train segmentation algorithms from labeled volumes.
- **Crop utility**: This module allows you to crop your volumes and labels dynamically, by selecting a fixed size volume and moving it around the image.

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [MIT] license,
"napari-cellseg-annotator" is free and open source software

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
