import os  # TODO(cyril): remove os
from pathlib import Path

import napari
import numpy as np
import scipy.ndimage as ndimage
from skimage.filters import threshold_otsu
from tifffile import imread, imwrite

from napari_cellseg3d.code_models.instance_segmentation import binary_watershed

# import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


"""
New code by Yves Paychere
Creates labels of artifacts in an image based on existing labels of neurons
"""


def map_labels(labels, artefacts):
    """Map the artefacts labels to the neurons labels.

    Parameters
    ----------
    labels : ndarray
        Label image with neurons labelled as mulitple values.
    artefacts : ndarray
        Label image with artefacts labelled as mulitple values.

    Returns:
    -------
    map_labels_existing: numpy array
        The label value of the artefact and the label value of the neurone associated or the neurons associated
    new_labels: list
        The labels of the artefacts that are not labelled in the neurons.
    """
    map_labels_existing = []
    new_labels = []

    for i in np.unique(artefacts):
        if i == 0:
            continue
        indexes = labels[artefacts == i]
        # find the most common label in the indexes
        unique, counts = np.unique(indexes, return_counts=True)
        unique = np.flip(unique[np.argsort(counts)])
        counts = np.flip(counts[np.argsort(counts)])
        if unique[0] != 0:
            map_labels_existing.append(
                np.array([i, unique[np.argmax(counts)]])
            )
        elif (
            counts[0] < np.sum(counts) * 2 / 3.0
        ):  # the artefact is connected to multiple neurons
            total = 0
            ii = 1
            while total < np.size(indexes) / 3.0:
                total = np.sum(counts[1 : ii + 1])
                ii += 1
            map_labels_existing.append(np.append([i], unique[1 : ii + 1]))
        else:
            new_labels.append(i)

    return map_labels_existing, new_labels


def make_labels(
    image,
    path_labels_out,
    threshold_factor=1,
    threshold_size=30,
    label_value=1,
    do_multi_label=True,
    use_watershed=True,
    augment_contrast_factor=2,
):
    """Detect nucleus. using a binary watershed algorithm and otsu thresholding.

    Parameters
    ----------
    image : str
        Path to image.
    path_labels_out : str
        Path of the output labelled image.
    threshold_size : int, optional
        Threshold for nucleus size, if the nucleus is smaller than this value it will be removed.
    label_value : int, optional
        Value to use for the label image.
    do_multi_label : bool, optional
        If True, each different nucleus will be labelled as a different value.
    use_watershed : bool, optional
        If True, use watershed algorithm to detect nucleus.
    augment_contrast_factor : int, optional
        Factor to augment the contrast of the image.

    Returns:
    -------
    ndarray
        Label image with nucleus labelled with 1 value per nucleus.
    """
    # image = imread(image)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))

    threshold_brightness = threshold_otsu(image) * threshold_factor
    image_contrasted = np.where(image > threshold_brightness, image, 0)

    if use_watershed:
        image_contrasted = (image_contrasted - np.min(image_contrasted)) / (
            np.max(image_contrasted) - np.min(image_contrasted)
        )
        image_contrasted = image_contrasted * augment_contrast_factor
        image_contrasted = np.where(image_contrasted > 1, 1, image_contrasted)
        labels = binary_watershed(image_contrasted, thres_small=threshold_size)
    else:
        labels = ndimage.label(image_contrasted)[0]

    labels = select_artefacts_by_size(
        labels, min_size=threshold_size, is_labeled=True
    )

    if not do_multi_label:
        labels = np.where(labels > 0, label_value, 0)

    imwrite(path_labels_out, labels.astype(np.uint16))
    imwrite(
        path_labels_out.replace(".tif", "_contrast.tif"),
        image_contrasted.astype(np.float32),
    )


def select_image_by_labels(image, labels, path_image_out, label_values):
    """Select image by labels.

    Parameters
    ----------
    image : np.array
        image.
    labels : np.array
        labels.
    path_image_out : str
        Path of the output image.
    label_values : list
        List of label values to select.
    """
    # image = imread(image)
    # labels = imread(labels)
    image = np.where(np.isin(labels, label_values), image, 0)
    imwrite(path_image_out, image.astype(np.float32))


# select the smallest cube that contains all the non-zero pixels of a 3d image
def get_bounding_box(img):
    height = np.any(img, axis=(0, 1))
    rows = np.any(img, axis=(0, 2))
    cols = np.any(img, axis=(1, 2))

    xmin, xmax = np.where(cols)[0][[0, -1]]
    ymin, ymax = np.where(rows)[0][[0, -1]]
    zmin, zmax = np.where(height)[0][[0, -1]]
    return xmin, xmax, ymin, ymax, zmin, zmax


# crop the image
def crop_image(img):
    xmin, xmax, ymin, ymax, zmin, zmax = get_bounding_box(img)
    return img[xmin:xmax, ymin:ymax, zmin:zmax]


def crop_image_path(image, path_image_out):
    """Crop image.

    Parameters
    ----------
    image : np.array
        image
    path_image_out : str
        Path of the output image.
    """
    image = crop_image(image)
    imwrite(path_image_out, image.astype(np.float32))


def make_artefact_labels(
    image,
    labels,
    threshold_artefact_brightness_percent=40,
    threshold_artefact_size_percent=1,
    contrast_power=20,
    label_value=2,
    do_multi_label=False,
    remove_true_labels=True,
):
    """Detect pseudo nucleus.

    Parameters
    ----------
    image : ndarray
        Image.
    labels : ndarray
        Label image.
    threshold_artefact_brightness_percent : int, optional
        Threshold for artefact brightness.
    threshold_artefact_size_percent : int, optional
        Threshold for artefact size, if the artefcact is smaller than this percentage of the neurons it will be removed.
    contrast_power : int, optional
        Power for contrast enhancement.
    label_value : int, optional
        Value to use for the label image.
    do_multi_label : bool, optional
        If True, each different artefact will be labelled as a different value.
    remove_true_labels : bool, optional
        If True, the true labels will be removed from the artefacts.

    Returns:
    -------
    ndarray
        Label image with pseudo nucleus labelled with 1 value per artefact.
    """
    neurons = np.array(labels > 0)
    non_neurons = np.array(labels == 0)

    image = (image - np.min(image)) / (np.max(image) - np.min(image))

    # calculate the percentile of the intensity of all the pixels that are labeled as neurons
    # check if the neurons are not empty
    if np.sum(neurons) > 0:
        threshold = np.percentile(
            image[neurons], threshold_artefact_brightness_percent
        )
    else:
        # take the percentile of the non neurons if the neurons are empty
        threshold = np.percentile(image[non_neurons], 90)

    # modify the contrast of the image accoring to the threshold with a tanh function and map the values to [0,1]

    image_contrasted = np.tanh((image - threshold) * contrast_power)
    image_contrasted = (image_contrasted - np.min(image_contrasted)) / (
        np.max(image_contrasted) - np.min(image_contrasted)
    )

    artefacts = binary_watershed(
        image_contrasted, thres_seeding=0.95, thres_small=15, thres_objects=0.4
    )

    if remove_true_labels:
        # evaluate where the artefacts are connected to the neurons
        # map the artefacts label to the neurons label
        map_labels_existing, new_labels = map_labels(labels, artefacts)

        # remove the artefacts that are connected to the neurons
        for i in map_labels_existing:
            artefacts[artefacts == i[0]] = 0
        # remove all the pixels of the neurons from the artefacts
        artefacts = np.where(labels > 0, 0, artefacts)

    # remove the artefacts that are too small
    # calculate the percentile of the size of the neurons
    if np.sum(neurons) > 0:
        sizes = ndimage.sum_labels(labels > 0, labels, np.unique(labels))
        neurone_size_percentile = np.percentile(
            sizes, threshold_artefact_size_percent
        )
    else:
        # find the size of each connected component
        sizes = ndimage.sum_labels(labels > 0, labels, np.unique(labels))
        # remove the smallest connected components
        neurone_size_percentile = np.percentile(sizes, 95)

    # select the artefacts that are bigger than the percentile

    artefacts = select_artefacts_by_size(
        artefacts, min_size=neurone_size_percentile, is_labeled=True
    )

    # relabel with the label value if the artefacts are not multi label
    if not do_multi_label:
        artefacts = np.where(artefacts > 0, label_value, artefacts)

    return artefacts


def select_artefacts_by_size(artefacts, min_size, is_labeled=False):
    """Select artefacts by size.

    Parameters
    ----------
    artefacts : ndarray
        Label image with artefacts labelled as 1.
    min_size : int, optional
        Minimum size of artefacts to keep
    is_labeled : bool, optional
        If True, the artefacts are already labelled.

    Returns:
    -------
    ndarray
        Label image with artefacts labelled and small artefacts removed.
    """
    labels = ndimage.label(artefacts)[0] if not is_labeled else artefacts

    # remove the small components
    labels_i, counts = np.unique(labels, return_counts=True)
    labels_i = labels_i[counts > min_size]
    labels_i = labels_i[labels_i > 0]
    return np.where(np.isin(labels, labels_i), labels, 0)


def create_artefact_labels(
    image,
    labels,
    output_path,
    threshold_artefact_brightness_percent=40,
    threshold_artefact_size_percent=1,
    contrast_power=20,
):
    """Create a new label image with artefacts labelled as 2 and neurons labelled as 1.

    Parameters
    ----------
    image : np.array
        image for artefact detection.
    labels : np.array
        label image array with each neurons labelled as a different int value.
    output_path : str
        Path to save the output label image file.
    threshold_artefact_brightness_percent : int, optional
        The artefacts need to be as least as bright as this percentage of the neurone's pixels.
    threshold_artefact_size : int, optional
        The artefacts need to be at least as big as this percentage of the neurons.
    contrast_power : int, optional
        Power for contrast enhancement.
    """
    artefacts = make_artefact_labels(
        image,
        labels,
        threshold_artefact_brightness_percent,
        threshold_artefact_size_percent,
        contrast_power=contrast_power,
        label_value=2,
        do_multi_label=False,
    )

    neurons_artefacts_labels = np.where(labels > 0, 1, artefacts)
    imwrite(output_path, neurons_artefacts_labels)


def visualize_images(paths):
    """Visualize images.

    Parameters
    ----------
    paths : list
        List of images to visualize.
    """
    viewer = napari.Viewer(ndisplay=3)
    for path in paths:
        image = imread(path)
        viewer.add_image(image)
    # wait for the user to close the viewer
    napari.run()


def create_artefact_labels_from_folder(
    path,
    do_visualize=False,
    threshold_artefact_brightness_percent=40,
    threshold_artefact_size_percent=1,
    contrast_power=20,
):
    """Create a new label image with artefacts labelled as 2 and neurons labelled as 1 for all images in a folder. The images created are stored in a folder artefact_neurons.

    Parameters
    ----------
    path : str
        Path to folder with images in folder volumes and labels in folder lab_sem. The images are expected to have the same alphabetical order in both folders.
    do_visualize : bool, optional
        If True, the images will be visualized.
    threshold_artefact_brightness_percent : int, optional
        The artefacts need to be as least as bright as this percentage of the neurone's pixels.
    threshold_artefact_size : int, optional
        The artefacts need to be at least as big as this percentage of the neurons.
    contrast_power : int, optional
        Power for contrast enhancement.
    """
    # find all the images in the folder and create a list
    path_labels = [
        f for f in os.listdir(path + "/labels") if f.endswith(".tif")
    ]
    path_images = [
        f for f in os.listdir(path + "/volumes") if f.endswith(".tif")
    ]
    # sort the list
    path_labels.sort()
    path_images.sort()
    # create the output folder
    Path().mkdir(path + "/artefact_neurons", exist_ok=True)
    # create the artefact labels
    for i in range(len(path_images)):
        print(path_labels[i])
        # consider that the images and the labels have names in the same alphabetical order
        create_artefact_labels(
            path + "/volumes/" + path_images[i],
            path + "/labels/" + path_labels[i],
            path + "/artefact_neurons/" + path_labels[i],
            threshold_artefact_brightness_percent,
            threshold_artefact_size_percent,
            contrast_power,
        )
        if do_visualize:
            visualize_images(
                [
                    path + "/volumes/" + path_images[i],
                    path + "/labels/" + path_labels[i],
                    path + "/artefact_neurons/" + path_labels[i],
                ]
            )


# if __name__ == "__main__":
#     repo_path = Path(__file__).resolve().parents[1]
#     print(f"REPO PATH : {repo_path}")
#     paths = [
#         "dataset_clean/cropped_visual/train",
#         "dataset_clean/cropped_visual/val",
#         "dataset_clean/somatomotor",
#         "dataset_clean/visual_tif",
#     ]
#     for data_path in paths:
#         path = str(repo_path / data_path)
#         print(path)
#         create_artefact_labels_from_folder(
#             path,
#             do_visualize=False,
#             threshold_artefact_brightness_percent=20,
#             threshold_artefact_size_percent=1,
#             contrast_power=20,
#         )
