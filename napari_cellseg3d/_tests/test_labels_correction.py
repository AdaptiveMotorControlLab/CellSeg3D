from pathlib import Path

import numpy as np
from tifffile import imread

from napari_cellseg3d.dev_scripts import artefact_labeling as al
from napari_cellseg3d.dev_scripts import correct_labels as cl
from napari_cellseg3d.dev_scripts import evaluate_labels as el

res_folder = Path(__file__).resolve().parent / "res"
image_path = res_folder / "test.tif"
image = imread(str(image_path))

labels_path = res_folder / "test_labels.tif"
labels = imread(str(labels_path))  # .astype(np.int32)


def test_artefact_labeling():
    output_path = str(res_folder / "test_artifacts.tif")
    al.create_artefact_labels(image, labels, output_path=output_path)
    assert Path(output_path).is_file()


def test_artefact_labeling_utils():
    crop_test = al.crop_image(image)
    assert isinstance(crop_test, np.ndarray)
    output_path = str(res_folder / "test_cropped.tif")
    al.crop_image_path(image, path_image_out=output_path)
    assert Path(output_path).is_file()


def test_correct_labels():
    output_path = res_folder / "test_correct"
    output_path.mkdir(exist_ok=True, parents=True)
    cl.relabel_non_unique_i(
        labels, str(output_path / "corrected.tif"), go_fast=True
    )


def test_relabel():
    cl.relabel(
        str(image_path),
        str(labels_path),
        go_fast=True,
        test=True,
    )


def test_evaluate_model_performance():
    el.evaluate_model_performance(
        labels, labels, print_details=True, visualize=False
    )
