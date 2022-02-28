"""
This module is an example of a barebones writer plugin for napari.

It implements the Writer specification.
see: https://napari.org/plugins/stable/guides.html#writers

Replace code below according to your needs.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, List, Any, Sequence, Tuple, Union

if TYPE_CHECKING:
    DataType = Union[Any, Sequence[Any]]
    FullLayerData = Tuple[DataType, dict, str]


def write_single_image(path: str, data: Any, meta: dict):
    """Writes a single image layer"""
    pass


def write_multiple(path: str, data: List[FullLayerData]):
    """Writes multiple layers of different types."""
    pass
