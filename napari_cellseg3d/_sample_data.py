"""Sample data contributed to napari's File > Open Sample menu."""

import hashlib
import urllib.request
from pathlib import Path

from napari_cellseg3d.utils import LOGGER as logger

C5IMAGE_NAME = "c5image.tif"
C5IMAGE_REPO_FILE = Path(__file__).parents[1] / "examples" / C5IMAGE_NAME
C5IMAGE_CACHE_FILE = Path(__file__).parent / "res" / "sample_data" / C5IMAGE_NAME
C5IMAGE_URL = (
    "https://raw.githubusercontent.com/AdaptiveMotorControlLab/"
    f"CellSeg3D/main/examples/{C5IMAGE_NAME}"
)
C5IMAGE_SHA256 = "9bb5a3da2ade6f66899da7247ca0b18c6c7adb43b9bd6b2c3869a32945a7d9ff"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _download(url: str, dest: Path, sha256: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    logger.info(f"Downloading sample data from {url}")
    try:
        with urllib.request.urlopen(url) as response:  # noqa: S310
            tmp.write_bytes(response.read())
        digest = _sha256(tmp)
        if digest != sha256:
            raise ValueError(
                f"Checksum mismatch for {url}: expected {sha256}, got {digest}"
            )
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)


def _c5image_path() -> Path:
    if C5IMAGE_REPO_FILE.is_file():
        return C5IMAGE_REPO_FILE
    if not C5IMAGE_CACHE_FILE.is_file():
        _download(C5IMAGE_URL, C5IMAGE_CACHE_FILE, C5IMAGE_SHA256)
    return C5IMAGE_CACHE_FILE


def c5image():
    """Small demo volume from IDR project 853 (see examples/README.md)."""
    from tifffile import imread

    return [(imread(_c5image_path()), {"name": "c5image"}, "image")]
