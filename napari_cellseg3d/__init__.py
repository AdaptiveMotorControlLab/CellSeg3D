"""napari-cellseg3d - napari plugin for cell segmentation."""

try:
    from napari_cellseg3d._version import version as __version__
except ImportError:
    from importlib.metadata import version

    __version__ = version("napari_cellseg3d")
