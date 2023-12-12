"""Utilities functions, classes, and variables."""
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Union

import napari
import numpy as np
import torch
from monai.transforms import Zoom
from numpy.random import PCG64, Generator
from tifffile import imread, imwrite

LOGGER = logging.getLogger(__name__)
###############
# Global logging level setting
# LOGGER.setLevel(logging.DEBUG)
LOGGER.setLevel(logging.INFO)
###############
"""
utils.py
====================================
Definitions of utility functions, classes, and variables
"""

rand_gen = Generator(PCG64(12345))


####################
# viewer utils
def save_folder(
    results_path, folder_name, images, image_paths, exist_ok=False
):
    """Saves a list of images in a folder.

    Args:
        results_path: Path to the folder containing results
        folder_name: Name of the folder containing results
        images: List of images to save
        image_paths: list of filenames of images
        exist_ok: whether to check for existing files. If False, will raise an error if the folder already exists.
    """
    results_folder = results_path / Path(folder_name)
    results_folder.mkdir(exist_ok=exist_ok, parents=True)

    for file, image in zip(image_paths, images):
        path = results_folder / Path(file).name

        imwrite(
            path,
            image,
        )
    LOGGER.info(f"Saved processed folder as : {results_folder}")


def save_layer(results_path, image_name, image):
    """Saves an image layer at the specified path.

    Args:
        results_path: path to folder containing result
        image_name: image name for saving
        image: data array containing image
    """
    path = str(results_path / Path(image_name))  # TODO flexible filetype
    LOGGER.info(f"Saved as : {path}")
    image = image.astype(np.float32)
    imwrite(path, image, dtype="float32")


def show_result(
    viewer,
    layer,
    image,
    name,
    existing_layer: napari.layers.Layer = None,
    colormap="bop orange",
    add_as_labels=False,
    add_as_image=False,
) -> napari.layers.Layer:
    """Adds layers to a viewer to show result to user.

    Args:
        viewer: viewer to add layer in
        layer: original layer the operation was run on, to determine whether it should be an Image or Labels layer
        image: the data array containing the image
        name: name of the added layer
        existing_layer: existing layer to update, if any
        colormap: colormap to use for the layer
        add_as_labels: whether to add the layer as a Labels layer. Overrides guessing from layer type.
        add_as_image: whether to add the layer as an Image layer. Overrides guessing from layer type.

    Returns:
        napari.layers.Layer: the layer added to the viewer
    """
    colormap = colormap if colormap is not None else "gray"
    if existing_layer is None:
        if add_as_image:
            LOGGER.info("Added resulting image layer")
            results_layer = viewer.add_image(
                image, name=name, colormap=colormap
            )
        elif add_as_labels:
            LOGGER.info("Added resulting label layer")
            results_layer = viewer.add_labels(image, name=name)
        else:
            if isinstance(layer, napari.layers.Image):
                LOGGER.info("Added resulting image layer")
                results_layer = viewer.add_image(
                    image, name=name, colormap=colormap
                )
            elif isinstance(layer, napari.layers.Labels):
                LOGGER.info("Added resulting label layer")
                results_layer = viewer.add_labels(image, name=name)
            else:
                LOGGER.warning(
                    f"Results not shown, unsupported layer type {type(layer)}"
                )
    else:
        try:
            viewer.layers[existing_layer.name].data = image
            results_layer = viewer.layers[existing_layer.name]
        except KeyError:
            LOGGER.warning(
                f"Results not shown, layer {existing_layer.name} not found"
                "Showing new layer instead"
            )
            results_layer = show_result(
                viewer, layer, image, name, existing_layer=None
            )
    return results_layer


class Singleton(type):
    """Singleton class that can only be instantiated once at a time, with said unique instance always being accessed on call.

    Should be used as a metaclass for classes without inheritance (object type).
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """Call method for Singleton class.

        Ensures that only one instance of the class is created at a time, and that it is always the same instance that is returned.
        """
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


def normalize_x(image):
    """Normalizes the values of an image array to be between [-1;1] rather than [0;255].

    Args:
        image (array): Image to process

    Returns:
        array: normalized value for the image
    """
    return image / 127.5 - 1


def mkdir_from_str(path: str, exist_ok=True, parents=True):
    """Creates a directory from a string path."""
    Path(path).resolve().mkdir(exist_ok=exist_ok, parents=parents)


def normalize_y(image):
    """Normalizes the values of an image array to be between [0;1] rather than [0;255].

    Args:
        image (array): Image to process

    Returns:
        array: normalized value for the image
    """
    return image / 255


def sphericity_volume_area(volume, surface_area):
    r"""Computes the sphericity from volume and area.

    .. math::
       sphericity =\\frac {\\pi^\\frac{1}{3} (6 V_{p})^\\frac{2}{3}} {A_p}

    """
    return (np.pi ** (1 / 3)) * ((6 * volume) ** (2 / 3)) / surface_area


def sphericity_axis(semi_major, semi_minor):
    r"""Computes the sphericity from volume semi major (a) and semi minor (b) axes.

    .. math::
        sphericity = \\frac {2 \\sqrt[3]{ab^2}} {a+ \\frac {b^2} {\\sqrt{a^2-b^2}}ln( \\frac {a+ \\sqrt{a^2-b^2}} {b} )}

    """
    a = semi_major
    b = semi_minor

    root = np.sqrt(a**2 - b**2)
    try:
        result = (
            2
            * (a * (b**2)) ** (1 / 3)
            / (a + (b**2) / root * np.log((a + root) / b))
        )
    except ZeroDivisionError:
        # LOGGER.debug(
        #     "Zero division in sphericity calculation was replaced by None"
        # )
        result = None
    except ValueError as e:
        LOGGER.warning(f"Error encountered in calculation : {e}")
        result = "Error in calculation"

    if math.isnan(result):
        # LOGGER.debug("NaN in sphericity calculation was replaced by None")
        result = None

    return result


def dice_coeff(
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred: Union[torch.Tensor, np.ndarray],
    smooth: float = 1.0,
) -> Union[torch.Tensor, np.float64]:
    """Compute Dice-Sorensen coefficient between two numpy arrays.

    Args:
        y_true: Ground truth label
        y_pred: Prediction label
        smooth: Smoothing factor to avoid division by zero

    Returns: dice coefficient.
    """
    if isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray):
        sum_tensor = np.sum
    elif isinstance(y_true, torch.Tensor) and isinstance(y_pred, torch.Tensor):
        sum_tensor = torch.sum
    else:
        raise ValueError(
            "y_true and y_pred must both be either numpy arrays or torch tensors"
        )

    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = sum_tensor(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        sum_tensor(y_true_f) + sum_tensor(y_pred_f) + smooth
    )


def seek_best_dice_coeff_channel(y_pred, y_true) -> torch.Tensor:
    """Compute Dice-Sorensen coefficient between unsupervised model output and ground truth labels; returns the channel with the highest dice coefficient.

    Args:
        y_true: Ground truth label
        y_pred: Prediction label
    Returns: best Dice coefficient channel.
    """
    dices = []
    # Find in which channel the labels are (to avoid background)
    for channel in range(y_pred.shape[1]):
        dices.append(
            dice_coeff(
                y_pred=y_pred[0, channel : (channel + 1), :, :, :],
                y_true=y_true[0],
            )
        )
    LOGGER.debug(f"DICE COEFF: {dices}")
    max_dice_channel = torch.argmax(torch.Tensor(dices))
    LOGGER.debug(f"MAX DICE CHANNEL: {max_dice_channel}")
    return max_dice_channel


def correct_rotation(image):
    """Rotates the axes 0 and 2 in [DHW] section of image array."""
    extra_dims = len(image.shape) - 3
    return np.swapaxes(image, 0 + extra_dims, 2 + extra_dims)


def normalize_max(image):
    """Normalizes an image using the max and min value."""
    shape = image.shape
    image = image.flatten()
    image = (image - image.min()) / (image.max() - image.min())
    image = image.reshape(shape)
    return image


def remap_image(
    image: Union["np.ndarray", "torch.Tensor"],
    new_max=100,
    new_min=0,
    prev_max=None,
    prev_min=None,
):
    """Normalizes a numpy array or Tensor using the max and min value."""
    shape = image.shape
    image = image.flatten()
    im_max = prev_max if prev_max is not None else image.max()
    im_min = prev_min if prev_min is not None else image.min()
    image = (image - im_min) / (im_max - im_min)
    image = image * (new_max - new_min) + new_min
    image = image.reshape(shape)
    return image


def resize(image, zoom_factors):
    """Resizes an image using the zoom_factors."""
    isotropic_image = Zoom(
        zoom_factors,
        keep_size=False,
        mode="nearest-exact",
        padding_mode="empty",
    )(np.expand_dims(image, axis=0))
    return isotropic_image[0].numpy()


def align_array_sizes(array_shape, target_shape):
    """Aligns the sizes of two arrays by adding zeros to the smaller one."""
    index_differences = []
    for i in range(len(target_shape)):
        if target_shape[i] != array_shape[i]:
            for j in range(len(array_shape)):
                if array_shape[i] == target_shape[j] and j != i:
                    index_differences.append({"origin": i, "target": j})

    # LOGGER.debug(index_differences)
    if len(index_differences) == 0:
        return [0, 1, 2], [-3, -2, -1]

    origins = []
    targets = []

    for diffs in index_differences:
        origins.append(diffs["origin"])
        targets.append(diffs["target"])

    reverse_mapping = {0: (-3), 1: (-2), 2: (-1)}
    for i in range(len(targets)):
        targets[i] = reverse_mapping[targets[i]]
    infos = np.unique(origins, return_index=True, return_counts=True)
    {"origins": infos[0], "index": infos[1], "counts": infos[2]}
    # LOGGER.debug(info_dict)

    final_orig = []
    final_targ = []
    for i in range(len(infos[0])):
        if infos[2][i] == 1:
            final_orig.append(infos[0][i])
            final_targ.append(targets[infos[1][i]])
    # LOGGER.debug(final_orig, final_targ)

    return final_orig, final_targ


def time_difference(time_start, time_finish, as_string=True):
    """Computes the time difference between two datetime objects.

    Args:
    time_start (datetime): time to subtract to time_finish
    time_finish (datetime): time to add to subtract time_start to
    as_string (bool): if True, returns a string with the full time diff. Otherwise, returns as a list [hours,minutes,seconds].
    """
    time_taken = time_finish - time_start
    days = divmod(time_taken.total_seconds(), 86400)  # Get days (without [0]!)
    hours = divmod(days[1], 3600)  # Use remainder of days to calc hours
    minutes = divmod(hours[1], 60)  # Use remainder of hours to calc minutes
    seconds = divmod(minutes[1], 1)  # Use remainder of minutes to calc seconds

    hours = f"{int(hours[0])}".zfill(2)
    minutes = f"{int(minutes[0])}".zfill(2)
    seconds = f"{int(seconds[0])}".zfill(2)

    return (
        f"{hours}:{minutes}:{seconds}"
        if as_string
        else [hours, minutes, seconds]
    )


def get_padding_dim(image_shape, anisotropy_factor=None):
    """Finds the nearest and superior power of two for each image dimension to zero-pad it for CNN processing.

    Accepts either 2D or 3D images shapes. E.g. an image size of 30x40x100 will result in a padding of 32x64x128.
    Shows a warning if the padding dimensions are very large.

    Args:
        image_shape (torch.size): an array of the dimensions of the image in D/H/W if 3D or H/W if 2D
        anisotropy_factor (list): anisotropy factor for each dimension

    Returns:
        array(int): padding value for each dim
    """
    padding = []

    dims = len(image_shape)
    LOGGER.debug(f"Data is {dims}D")
    LOGGER.debug(f"Image shape is {image_shape}")
    if dims != 2 and dims != 3:
        error = "Please check the dimensions of the input, only 2 or 3-dimensional data is supported currently"
        LOGGER.error(error)
        raise ValueError(error)

    for i in range(dims):
        n = 0
        pad = -1
        size = image_shape[i]
        if anisotropy_factor is not None:
            # problems with zero divs avoided via params for spinboxes
            size = int(size / anisotropy_factor[i])
        while pad < size:
            # if size - pad < 30:
            #     LOGGER.warning(
            #         f"Your value is close to a lower power of two; you might want to choose slightly smaller"
            #         f" sizes and/or crop your images down to {pad}"
            #     )

            pad = 2**n
            n += 1
            if pad >= 1024:
                LOGGER.warning(
                    "Warning : a very large dimension for automatic padding has been computed.\n"
                    "Ensure your images are of an appropriate size and/or that you have enough memory."
                    f"The padding value is currently {pad}."
                )

        padding.append(pad)

    LOGGER.debug(f"Padding sizes are {padding}")
    return padding


def denormalize_y(image):
    """De-normalizes the values of an image array to be between [0;255] rather than [0;1].

    Args:
        image (array): Image to process

    Returns:
        array: de-normalized value for the image
    """
    return image * 255


def fill_list_in_between(lst, n, fill_value):
    """Fills a list with n * elem between each member of list.

    Example with list = [1,2,3], n=2, elem='&' : returns [1, &, &,2,&,&,3,&,&].

    Args:
        lst: list to fill
        n: number of elements to add
        fill_value: added n times after each element of list

    Returns :
        Filled list
    """
    new_list = []
    for i in range(len(lst)):
        temp_list = [lst[i]]
        while len(temp_list) < n + 1:
            temp_list.append(fill_value)
        if i < len(lst) - 1:
            new_list += temp_list
        else:
            new_list.append(lst[i])
            for _j in range(n):
                new_list.append(fill_value)
            return new_list
    return None


def parse_default_path(possible_paths, check_existence=True):
    """Returns a default path based on a vector of paths, some of which might be empty.

    Args:
        possible_paths: array of paths
        check_existence: whether to check if the path exists.

    Returns: the chosen default path

    """
    default_paths = []
    LOGGER.debug(f"Utils : possible paths are {possible_paths}")
    if any(path is not None for path in possible_paths):
        for p in possible_paths:
            if p is not None and (Path(p).exists() or not check_existence):
                default_paths.append(str(Path(p).resolve().as_posix()))
        # default_paths = [
        #     str(Path(p).resolve()) for p in possible_paths if (Path(p).exists())
        # ]
    # default_paths = [
    #     path for path in default_paths if path is not None and path != []
    # ]
    LOGGER.debug(f"Utils : default paths are {default_paths}")
    if len(default_paths) == 0:
        return str(Path().home())
    default_path = max(default_paths, key=len)
    return str(default_path)


def get_date_time():
    """Get date and time in the following format : year_month_day_hour_minute_second."""
    return f"{datetime.now():%Y_%m_%d_%H_%M_%S}"


def get_time():
    """Get time in the following format : hour:minute:second. NOT COMPATIBLE with file paths (saving with ":" is invalid)."""
    return f"{datetime.now():%H:%M:%S}"


def get_time_filepath():
    """Get time in the following format : hour_minute_second. Compatible with saving."""
    return f"{datetime.now():%H_%M_%S}"


def get_all_matching_files(path, pattern=None):
    """Returns a list of all files in a directory matching the pattern.

    Args:
        path (str): path to the directory containing the images
        pattern (list): list of file extensions to match, defaults to [".tif", ".tiff"] if None

    Returns:
        list: list of all files in the directory matching the pattern, sorted alphabetically
    """
    if pattern is None:
        pattern = {".tif", ".tiff"}
    path = Path(path).resolve()
    if not path.exists():
        raise ValueError(f"Data folder {path} does not exist")
    files = list([p for p in Path(path).glob("*") if p.suffix in pattern])
    # for p in Path(path).glob("*"):
    #     LOGGER.debug(f"Found file {p} with suffix {p.suffix}")
    #     LOGGER.debug(f"Suffix in pattern : {p.suffix in pattern}")
    #     if p.suffix in pattern:
    #         LOGGER.debug(f"File {p} matches pattern")
    LOGGER.debug(f"Found files : {files}")
    try:
        if len(files) == 0:
            LOGGER.warning(f"No files found in {path}")
            return None
    except TypeError:
        if files is None:
            LOGGER.warning(f"No files found in {path}")
            return None
    return sorted(files)


def load_images(dir_or_path, filetype="", as_folder: bool = False):
    """Loads the images in ``directory``, with different behaviour depending on ``filetype`` and ``as_folder``.

    * If ``as_folder`` is **False**, will load the path as a single 3D **.tif** image.
    * If **True**, it will try to load a folder as stack of images. In this case ``filetype`` must be specified.

    If **True** :

        * For ``filetype == ".tif"`` : loads all tif files in the folder as a 3D dataset.

        * For  ``filetype == ".png"`` : loads all png files in the folder as a 3D dataset.


    Args:
        dir_or_path (str): path to the directory containing the images or the images themselves
        filetype (str): expected file extension of the image(s) in the directory, if as_folder is False
        as_folder (bool): Whether to load a folder of images as stack or a single 3D image

    Returns:
        np.array: array with loaded images
    """
    # if not as_folder:
    filename_pattern_original = Path(dir_or_path)
    return imread(str(filename_pattern_original))  # tifffile imread


def quantile_normalization(
    image: Union[np.ndarray, torch.Tensor],
    quantile_high=0.99,
    quantile_low=0.01,
):
    """Normalizes an image using the quantiles."""
    if quantile_high < quantile_low:
        raise ValueError(
            f"quantile_high must be greater than quantile_low, got {quantile_high} and {quantile_low}"
        )

    if isinstance(image, torch.Tensor):
        qtl = torch.quantile
        where = torch.where
    elif isinstance(image, np.ndarray):
        qtl = np.quantile
        where = np.where
    else:
        raise TypeError("image needs to be torch tensor or numpy array")

    shape = image.shape
    image = image.flatten()
    high_quantile_value = qtl(image, quantile_high)
    low_quantile_value = qtl(image, quantile_low)
    image = where(image > high_quantile_value, high_quantile_value, image)
    image = where(image < low_quantile_value, low_quantile_value, image)
    return image.reshape(shape)


def channels_fraction_above_threshold(volume: np.array, threshold=0.5) -> list:
    """Computes the fraction of pixels above a certain value in a 4D volume for each channel.

    Args:
        volume (np.ndarray): Array of shape (C, H, W, D) containing the input volume
        threshold (float): Threshold value to use for the computation

    Returns:
        list: List of length C containing the fraction of pixels above the threshold for each channel
    """
    if len(volume.shape) != 4:
        raise ValueError(
            f"Volume shape {volume.shape} is not 4D. Expecting CxHxWxD."
        )
    fractions = []
    for _i, channel in enumerate(volume):
        fractions.append(
            fraction_above_threshold(channel, threshold=threshold)
        )
    return fractions


def fraction_above_threshold(volume: np.array, threshold=0.5) -> float:
    """Computes the fraction of pixels above a certain value in a volume.

    Args:
        volume (np.ndarray): Array containing the input volume
        threshold (float): Threshold value to use for the computation.

    Returns:
        float: Fraction of pixels above the threshold
    """
    flattened = volume.flatten()
    above_thresh = np.where(flattened > threshold, 1, 0)
    LOGGER.debug(
        f"non zero in above_thresh : {np.count_nonzero(above_thresh)}"
    )
    return np.count_nonzero(above_thresh) / np.size(flattened)
