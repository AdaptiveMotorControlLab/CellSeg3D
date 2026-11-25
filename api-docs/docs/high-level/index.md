# High-level API

Entry points for running CellSeg3D from Python without the napari interface, as used in the [Colab notebooks](https://github.com/AdaptiveMotorControlLab/CellSeg3D/tree/main/notebooks).

- [Inference](inference.md): segment a volume with a pretrained model, then threshold and split it into instances.
- [Training](training.md): train WNet3D (self-supervised) or a supervised model.

Every object is configured with a dataclass from \[`napari_cellseg3d.config`\][napari_cellseg3d.config].

Other functionality is documented in the [API Reference](../reference/napari_cellseg3d/index.md).
