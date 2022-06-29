import importlib.util
import os
import warnings
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from dask_image.imread import imread as dask_imread
from pandas import DataFrame
from pandas import Series
from skimage import io
from skimage.filters import gaussian
from tifffile import imread as tfl_imread
from tqdm import tqdm

"""
utils.py
====================================
Definitions of utility functions and variables
"""


def normalize_x(image):
    """Normalizes the values of an image array to be between [-1;1] rather than [0;255]

    Args:
        image (array): Image to process

    Returns:
        array: normalized value for the image
    """
    image = image / 127.5 - 1
    return image


def normalize_y(image):
    """Normalizes the values of an image array to be between [0;1] rather than [0;255]

    Args:
        image (array): Image to process

    Returns:
        array: normalized value for the image
    """
    image = image / 255
    return image


def sphericity_volume_area(volume, surface_area):
    """Computes the sphericity from volume and area

    .. math::
       sphericity =\\frac {\\pi^\\frac{1}{3} (6 V_{p})^\\frac{2}{3}} {A_p}

    """
    return np.pi ** (1 / 3) * (6 * volume) ** (2 / 3) / surface_area


def sphericity_axis(semi_major, semi_minor):
    """Computes the sphericity from volume semi major (a) and semi minor (b) axes.

    .. math::
        sphericity = \\frac {2 \\sqrt[3]{ab^2}} {a+ \\frac {b^2} {\\sqrt{a^2-b^2}}ln( \\frac {a+ \\sqrt{a^2-b^2}} {b} )}

    """
    a = semi_major
    b = semi_minor

    root = (a**2 - b**2) ** (1 / 2)
    try:
        result = (
            2
            * (a * (b**2)) ** (1 / 3)
            / (a + (b**2) / root * np.log((a + root) / b))
        )
    except ZeroDivisionError:
        print("Zero division in sphericity calculation was replaced by 0")
        result = 0

    return result


def dice_coeff(y_true, y_pred):
    """Compute Dice-Sorensen coefficient between two numpy arrays

    Args:
        y_true: Ground truth label
        y_pred: Prediction label

    Returns: dice coefficient

    """
    smooth = 1.0
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    score = (2.0 * intersection + smooth) / (
        np.sum(y_true_f) + np.sum(y_pred_f) + smooth
    )
    return score


def align_array_sizes(array_shape, target_shape):

    index_differences = []
    for i in range(len(target_shape)):
        if target_shape[i] != array_shape[i]:
            for j in range(len(array_shape)):
                if array_shape[i] == target_shape[j]:
                    if j != i:
                        index_differences.append({"origin": i, "target": j})

    # print(index_differences)
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
    info_dict = {"origins": infos[0], "index": infos[1], "counts": infos[2]}
    # print(info_dict)

    final_orig = []
    final_targ = []
    for i in range(len(infos[0])):
        if infos[2][i] == 1:
            final_orig.append(infos[0][i])
            final_targ.append(targets[infos[1][i]])
    # print(final_orig, final_targ)

    return final_orig, final_targ


def time_difference(time_start, time_finish, as_string=True):
    """
    Args:
        time_start (datetime): time to subtract to time_finish
        time_finish (datetime): time to add to (-time_start)
        as_string (bool): if True, returns a string with the full time diff. Otherwise, returns as a list [hours,minutes,seconds]
    """

    time_taken = time_finish - time_start
    days = divmod(time_taken.total_seconds(), 86400)  # Get days (without [0]!)
    hours = divmod(days[1], 3600)  # Use remainder of days to calc hours
    minutes = divmod(hours[1], 60)  # Use remainder of hours to calc minutes
    seconds = divmod(minutes[1], 1)  # Use remainder of minutes to calc seconds

    hours = f"{int(hours[0])}".zfill(2)
    minutes = f"{int(minutes[0])}".zfill(2)
    seconds = f"{int(seconds[0])}".zfill(2)

    if as_string:
        return f"{hours}:{minutes}:{seconds}"
    else:
        return [hours, minutes, seconds]


def get_padding_dim(image_shape, anisotropy_factor=None):
    """
    Finds the nearest and superior power of two for each image dimension to zero-pad it for CNN processing,
    accepts either 2D or 3D images shapes. E.g. an image size of 30x40x100 will result in a padding of 32x64x128.
    Shows a warning if the padding dimensions are very large.

    Args:
        image_shape (torch.size): an array of the dimensions of the image in D/H/W if 3D or H/W if 2D

    Returns:
        array(int): padding value for each dim
    """
    padding = []

    dims = len(image_shape)
    print(f"Dimension of data for padding : {dims}D")
    print(f"Image shape is {image_shape}")
    if dims != 2 and dims != 3:
        error = "Please check the dimensions of the input, only 2 or 3-dimensional data is supported currently"
        print(error)
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
            #     warnings.warn(
            #         f"Your value is close to a lower power of two; you might want to choose slightly smaller"
            #         f" sizes and/or crop your images down to {pad}"
            #     )

            pad = 2**n
            n += 1
            if pad >= 256:
                warnings.warn(
                    "Warning : a very large dimension for automatic padding has been computed.\n"
                    "Ensure your images are of an appropriate size and/or that you have enough memory."
                    f"The padding value is currently {pad}."
                )

        padding.append(pad)

    print(f"Padding sizes are {padding}")
    return padding


def anisotropy_zoom_factor(resolutions):
    """Computes a zoom factor to correct anisotropy, based on resolutions

        Args:
            resolutions: array for resolution (float) in microns for each axis

    ith    Returns: an array with the corresponding zoom factors for each axis (all values divided by min)

    """

    base = min(resolutions)
    zoom_factors = [base / res for res in resolutions]
    return zoom_factors


def denormalize_y(image):
    """De-normalizes the values of an image array to be between [0;255] rather than [0;1]

    Args:
        image (array): Image to process

    Returns:
        array: de-normalized value for the image
    """
    return image * 255


def annotation_to_input(label_ermito):
    mito = (label_ermito == 1) * 255
    er = (label_ermito == 2) * 255
    mito = normalize_y(mito)
    er = normalize_y(er)
    mito_anno = np.zeros_like(mito)
    er_anno = np.zeros_like(er)
    mito = gaussian(mito, sigma=2) * 255
    er = gaussian(er, sigma=2) * 255
    mito_anno[:, :] = mito
    er_anno[:, :] = er
    anno = np.concatenate(
        [mito_anno[:, :, np.newaxis], er_anno[:, :, np.newaxis]], 2
    )
    anno = normalize_x(anno[np.newaxis, :, :, :])
    return anno


def check_csv(project_path, ext):
    if not os.path.isfile(
        os.path.join(project_path, os.path.basename(project_path) + ".csv")
    ):
        cols = [
            "project",
            "type",
            "ext",
            "z",
            "y",
            "x",
            "z_size",
            "y_size",
            "x_size",
            "created_date",
            "update_date",
            "path",
            "notes",
        ]
        df = DataFrame(index=[], columns=cols)
        filename_pattern_original = os.path.join(
            project_path, f"dataset/Original_size/Original/*{ext}"
        )
        images_original = dask_imread(filename_pattern_original)
        z, y, x = images_original.shape
        record = Series(
            [
                os.path.basename(project_path),
                "dataset",
                ".tif",
                0,
                0,
                0,
                z,
                y,
                x,
                datetime.datetime.now(),
                "",
                os.path.join(project_path, "dataset/Original_size/Original"),
                "",
            ],
            index=df.columns,
        )
        df = df.append(record, ignore_index=True)
        df.to_csv(
            os.path.join(project_path, os.path.basename(project_path) + ".csv")
        )
    else:
        pass


def check_annotations_dir(project_path):
    if not os.path.isdir(
        os.path.join(project_path, "annotations/Original_size/master")
    ):
        os.makedirs(
            os.path.join(project_path, "annotations/Original_size/master")
        )
    else:
        pass


def fill_list_in_between(lst, n, elem):
    """Fills a list with n * elem between each member of list.
    Example with list = [1,2,3], n=2, elem='&' : returns [1, &, &,2,&,&,3,&,&]

    Args:
        lst: list to fill
        n: number of elements to add
        elem: added n times after each element of list

    Returns :
        Filled list
    """

    new_list = []
    for i in range(len(lst)):
        temp_list = [lst[i]]
        while len(temp_list) < n + 1:
            temp_list.append(elem)
        if i < len(lst) - 1:
            new_list += temp_list
        else:
            new_list.append(lst[i])
            for j in range(n):
                new_list.append(elem)
            return new_list


def check_zarr(project_path, ext):
    if not len(
        list(
            (Path(project_path) / "dataset" / "Original_size").glob("./*.zarr")
        )
    ):
        filename_pattern_original = os.path.join(
            project_path, f"dataset/Original_size/Original/*{ext}"
        )
        images_original = dask_imread(filename_pattern_original)
        images_original.to_zarr(
            os.path.join(project_path, f"dataset/Original_size/Original.zarr")
        )
    else:
        pass


def check(project_path, ext):
    check_csv(project_path, ext)
    check_zarr(project_path, ext)
    check_annotations_dir(project_path)


def parse_default_path(possible_paths):
    """Returns a default path based on a vector of paths, some of which might be empty.

    Args:
        possible_paths: array of paths

    Returns: the chosen default path

    """

    # print("paths :")
    # print(default_paths)
    # print(default_path)

    default_paths = [
        p for p in possible_paths if (p != "" and p != [""] and len(p) >= 3)
    ]
    if len(default_paths) == 0:
        default_path = os.path.expanduser("~")
    else:
        default_path = max(default_paths)
    return default_path


def get_date_time():
    """Get date and time in the following format : year_month_day_hour_minute_second"""
    return "{:%Y_%m_%d_%H_%M_%S}".format(datetime.now())


def get_time():
    """Get time in the following format : hour:minute:second. NOT COMPATIBLE with file paths (saving with ":" is invalid)"""
    return "{:%H:%M:%S}".format(datetime.now())


def get_time_filepath():
    """Get time in the following format : hour_minute_second. Compatible with saving"""
    return "{:%H_%M_%S}".format(datetime.now())


def load_images(dir_or_path, filetype="", as_folder: bool = False):
    """Loads the images in ``directory``, with different behaviour depending on ``filetype`` and ``as_folder``

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
        dask.array.Array: dask array with loaded images
    """

    if not as_folder:
        filename_pattern_original = os.path.join(dir_or_path)
        # print(filename_pattern_original)
    elif as_folder and filetype != "":
        filename_pattern_original = os.path.join(dir_or_path + "/*" + filetype)
        # print(filename_pattern_original)
    else:
        raise ValueError("If loading as a folder, filetype must be specified")

    if as_folder:
        images_original = dask_imread(filename_pattern_original)
    else:
        images_original = tfl_imread(
            filename_pattern_original
        )  # tifffile imread

    return images_original


# def load_predicted_masks(mito_mask_dir, er_mask_dir, filetype):
#
#     images_mito_label = load_images(mito_mask_dir, filetype)
#     # TODO : check that there is no problem with compute when loading as single file
#     images_mito_label = images_mito_label.compute()
#     images_er_label = load_images(er_mask_dir, filetype)
#     # TODO : check that there is no problem with compute when loading as single file
#     images_er_label = images_er_label.compute()
#     base_label = (images_mito_label > 127) * 1 + (images_er_label > 127) * 2
#     return base_label


def load_saved_masks(mod_mask_dir, filetype, as_folder: bool):
    images_label = load_images(mod_mask_dir, filetype, as_folder)
    if as_folder:
        images_label = images_label.compute()
    base_label = images_label
    return base_label


def load_raw_masks(raw_mask_dir, filetype):
    images_raw = load_images(raw_mask_dir, filetype)
    # TODO : check that there is no problem with compute when loading as single file
    images_raw = images_raw.compute()
    base_label = np.where((126 < images_raw) & (images_raw < 171), 255, 0)
    return base_label


def save_stack(images, out_path, filetype=".png", check_warnings=False):
    """Saves the files in labels at location out_path as a stack of len(labels) .png files

    Args:
        images: array of label images
        out_path: path to the directory for saving
    """
    num = images.shape[0]
    os.makedirs(out_path, exist_ok=True)
    for i in range(num):
        label = images[i]
        io.imsave(
            os.path.join(out_path, str(i).zfill(4) + filetype),
            label,
            check_contrast=check_warnings,
        )


def load_X_gray(folder_path):
    image_files = []
    for file in os.listdir(folder_path):
        base, ext = os.path.splitext(file)
        if ext == ".png":
            image_files.append(file)
        else:
            pass

    image_files.sort()

    img = cv2.imread(
        folder_path + os.sep + image_files[0], cv2.IMREAD_GRAYSCALE
    )

    images = np.zeros(
        (len(image_files), img.shape[0], img.shape[1], 1), np.float32
    )
    for i, image_file in tqdm(enumerate(image_files)):
        image = cv2.imread(
            folder_path + os.sep + image_file, cv2.IMREAD_GRAYSCALE
        )
        image = image[:, :, np.newaxis]
        images[i] = normalize_x(image)

    print(images.shape)

    return images, image_files


def load_Y_gray(folder_path, thresh=None, normalize=False):
    image_files = []
    for file in os.listdir(folder_path):
        base, ext = os.path.splitext(file)
        if ext == ".png":
            image_files.append(file)
        else:
            pass

    image_files.sort()

    img = cv2.imread(
        folder_path + os.sep + image_files[0], cv2.IMREAD_GRAYSCALE
    )

    images = np.zeros(
        (len(image_files), img.shape[0], img.shape[1], 1), np.float32
    )

    for i, image_file in tqdm(enumerate(image_files)):
        image = cv2.imread(
            folder_path + os.sep + image_file, cv2.IMREAD_GRAYSCALE
        )
        if thresh:
            ret, image = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)
        image = image[:, :, np.newaxis]
        if normalize:
            images[i] = normalize_y(image)
        else:
            images[i] = image

    print(images.shape)

    return images, image_files


def select_train_data(dataframe, ori_imgs, label_imgs, ori_filenames):
    train_img_names = list()
    for node in dataframe.itertuples():
        if node.train == "Checked":
            train_img_names.append(node.filename)

    train_ori_imgs = list()
    train_label_imgs = list()
    for ori_img, label_img, train_filename in zip(
        ori_imgs, label_imgs, ori_filenames
    ):
        if train_filename in train_img_names:
            train_ori_imgs.append(ori_img)
            train_label_imgs.append(label_img)

    return np.array(train_ori_imgs), np.array(train_label_imgs)


def format_Warning(message, category, filename, lineno, line=""):
    """Formats a warning message, use in code with ``warnings.formatwarning = utils.format_Warning``

    Args:
        message: warning message
        category: which type of warning has been raised
        filename: file
        lineno: line number
        line: unused

    Returns: format

    """
    return (
        str(filename)
        + ":"
        + str(lineno)
        + ": "
        + category.__name__
        + ": "
        + str(message)
        + "\n"
    )


# def dice_coeff(y_true, y_pred):
#     smooth = 1.
#     y_true_f = y_true.flatten()
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
#     return score


# def dice_loss(y_true, y_pred):
#     loss = 1 - dice_coeff(y_true, y_pred)
#     return loss


# def bce_dice_loss(y_true, y_pred):
#     loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
#     return loss


def divide_imgs(images):
    H = -(-images.shape[1] // 412)
    W = -(-images.shape[2] // 412)

    diveded_imgs = np.zeros((images.shape[0] * H * W, 512, 512, 1), np.float32)
    # print(H, W)

    for z in range(images.shape[0]):
        image = images[z]
        for h in range(H):
            for w in range(W):
                cropped_img = np.zeros((512, 512, 1), np.float32)
                cropped_img -= 1

                if images.shape[1] < 412:
                    h = -1
                if images.shape[2] < 412:
                    w = -1

                if h == -1:
                    if w == -1:
                        cropped_img[
                            50 : images.shape[1] + 50,
                            50 : images.shape[2] + 50,
                            0,
                        ] = image[0 : images.shape[1], 0 : images.shape[2], 0]
                    elif w == 0:
                        cropped_img[
                            50 : images.shape[1] + 50, 50:512, 0
                        ] = image[0 : images.shape[1], 0:462, 0]
                    elif w == W - 1:
                        cropped_img[
                            50 : images.shape[1] + 50,
                            0 : images.shape[2] - 412 * W - 50,
                            0,
                        ] = image[
                            0 : images.shape[1],
                            w * 412 - 50 : images.shape[2],
                            0,
                        ]
                    else:
                        cropped_img[50 : images.shape[1] + 50, :, 0] = image[
                            0 : images.shape[1],
                            w * 412 - 50 : (w + 1) * 412 + 50,
                            0,
                        ]
                elif h == 0:
                    if w == -1:
                        cropped_img[
                            50:512, 50 : images.shape[2] + 50, 0
                        ] = image[0:462, 0 : images.shape[2], 0]
                    elif w == 0:
                        cropped_img[50:512, 50:512, 0] = image[0:462, 0:462, 0]
                    elif w == W - 1:
                        cropped_img[
                            50:512, 0 : images.shape[2] - 412 * W - 50, 0
                        ] = image[0:462, w * 412 - 50 : images.shape[2], 0]
                    else:
                        # cropped_img[50:512, :, 0] = image[0:462, w*412-50:(w+1)*412+50, 0]
                        try:
                            cropped_img[50:512, :, 0] = image[
                                0:462, w * 412 - 50 : (w + 1) * 412 + 50, 0
                            ]
                        except:
                            cropped_img[
                                50:512,
                                0 : images.shape[2] - 412 * (W - 1) - 50,
                                0,
                            ] = image[
                                0:462, w * 412 - 50 : (w + 1) * 412 + 50, 0
                            ]
                elif h == H - 1:
                    if w == -1:
                        cropped_img[
                            0 : images.shape[1] - 412 * H - 50,
                            50 : images.shape[2] + 50,
                            0,
                        ] = image[
                            h * 412 - 50 : images.shape[1],
                            0 : images.shape[2],
                            0,
                        ]
                    elif w == 0:
                        cropped_img[
                            0 : images.shape[1] - 412 * H - 50, 50:512, 0
                        ] = image[h * 412 - 50 : images.shape[1], 0:462, 0]
                    elif w == W - 1:
                        cropped_img[
                            0 : images.shape[1] - 412 * H - 50,
                            0 : images.shape[2] - 412 * W - 50,
                            0,
                        ] = image[
                            h * 412 - 50 : images.shape[1],
                            w * 412 - 50 : images.shape[2],
                            0,
                        ]
                    else:
                        try:
                            cropped_img[
                                0 : images.shape[1] - 412 * H - 50, :, 0
                            ] = image[
                                h * 412 - 50 : images.shape[1],
                                w * 412 - 50 : (w + 1) * 412 + 50,
                                0,
                            ]
                        except:
                            cropped_img[
                                0 : images.shape[1] - 412 * H - 50,
                                0 : images.shape[2] - 412 * (W - 1) - 50,
                                0,
                            ] = image[
                                h * 412 - 50 : images.shape[1],
                                w * 412 - 50 : (w + 1) * 412 + 50,
                                0,
                            ]
                else:
                    if w == -1:
                        cropped_img[:, 50 : images.shape[2] + 50, 0] = image[
                            h * 412 - 50 : (h + 1) * 412 + 50,
                            0 : images.shape[2],
                            0,
                        ]
                    elif w == 0:
                        # cropped_img[:, 50:512, 0] = image[h*412-50:(h+1)*412+50, 0:462, 0]
                        try:
                            cropped_img[:, 50:512, 0] = image[
                                h * 412 - 50 : (h + 1) * 412 + 50, 0:462, 0
                            ]
                        except:
                            cropped_img[
                                0 : images.shape[1] - 412 * H - 50 + 412,
                                50:512,
                                0,
                            ] = image[
                                h * 412 - 50 : (h + 1) * 412 + 50, 0:462, 0
                            ]
                    elif w == W - 1:
                        # cropped_img[:, 0:images.shape[2]-412*W-50, 0] = image[h*412-50:(h+1)*412+50, w*412-50:images.shape[2], 0]
                        try:
                            cropped_img[
                                :, 0 : images.shape[2] - 412 * W - 50, 0
                            ] = image[
                                h * 412 - 50 : (h + 1) * 412 + 50,
                                w * 412 - 50 : images.shape[2],
                                0,
                            ]
                        except:
                            cropped_img[
                                0 : images.shape[1] - 412 * H - 50 + 412,
                                0 : images.shape[2] - 412 * W - 50,
                                0,
                            ] = image[
                                h * 412 - 50 : (h + 1) * 412 + 50,
                                w * 412 - 50 : images.shape[2],
                                0,
                            ]
                    else:
                        # cropped_img[:, :, 0] = image[h*412-50:(h+1)*412+50, w*412-50:(w+1)*412+50, 0]
                        try:
                            cropped_img[:, :, 0] = image[
                                h * 412 - 50 : (h + 1) * 412 + 50,
                                w * 412 - 50 : (w + 1) * 412 + 50,
                                0,
                            ]
                        except:
                            try:
                                cropped_img[
                                    :,
                                    0 : images.shape[2] - 412 * (W - 1) - 50,
                                    0,
                                ] = image[
                                    h * 412 - 50 : (h + 1) * 412 + 50,
                                    w * 412 - 50 : (w + 1) * 412 + 50,
                                    0,
                                ]
                            except:
                                cropped_img[
                                    0 : images.shape[1] - 412 * (H - 1) - 50,
                                    :,
                                    0,
                                ] = image[
                                    h * 412 - 50 : (h + 1) * 412 + 50,
                                    w * 412 - 50 : (w + 1) * 412 + 50,
                                    0,
                                ]
                h = max(0, h)
                w = max(0, w)
                diveded_imgs[z * H * W + w * H + h] = cropped_img
                # print(z*H*W+ w*H+h)

    return diveded_imgs


def merge_imgs(imgs, original_image_shape):
    merged_imgs = np.zeros(
        (
            original_image_shape[0],
            original_image_shape[1],
            original_image_shape[2],
            1,
        ),
        np.float32,
    )
    H = -(-original_image_shape[1] // 412)
    W = -(-original_image_shape[2] // 412)

    for z in range(original_image_shape[0]):
        for h in range(H):
            for w in range(W):

                if original_image_shape[1] < 412:
                    h = -1
                if original_image_shape[2] < 412:
                    w = -1

                # print(z*H*W+ max(w, 0)*H+max(h, 0))
                if h == -1:
                    if w == -1:
                        merged_imgs[
                            z,
                            0 : original_image_shape[1],
                            0 : original_image_shape[2],
                            0,
                        ] = imgs[z * H * W + 0 * H + 0][
                            50 : original_image_shape[1] + 50,
                            50 : original_image_shape[2] + 50,
                            0,
                        ]
                    elif w == 0:
                        merged_imgs[
                            z, 0 : original_image_shape[1], 0:412, 0
                        ] = imgs[z * H * W + w * H + 0][
                            50 : original_image_shape[1] + 50, 50:462, 0
                        ]
                    elif w == W - 1:
                        merged_imgs[
                            z,
                            0 : original_image_shape[1],
                            w * 412 : original_image_shape[2],
                            0,
                        ] = imgs[z * H * W + w * H + 0][
                            50 : original_image_shape[1] + 50,
                            50 : original_image_shape[2] - 412 * W - 50,
                            0,
                        ]
                    else:
                        merged_imgs[
                            z,
                            0 : original_image_shape[1],
                            w * 412 : (w + 1) * 412,
                            0,
                        ] = imgs[z * H * W + w * H + 0][
                            50 : original_image_shape[1] + 50, 50:462, 0
                        ]
                elif h == 0:
                    if w == -1:
                        merged_imgs[
                            z, 0:412, 0 : original_image_shape[2], 0
                        ] = imgs[z * H * W + 0 * H + h][
                            50:462, 50 : original_image_shape[2] + 50, 0
                        ]
                    elif w == 0:
                        merged_imgs[z, 0:412, 0:412, 0] = imgs[
                            z * H * W + w * H + h
                        ][50:462, 50:462, 0]
                    elif w == W - 1:
                        merged_imgs[
                            z, 0:412, w * 412 : original_image_shape[2], 0
                        ] = imgs[z * H * W + w * H + h][
                            50:462,
                            50 : original_image_shape[2] - 412 * W - 50,
                            0,
                        ]
                    else:
                        merged_imgs[
                            z, 0:412, w * 412 : (w + 1) * 412, 0
                        ] = imgs[z * H * W + w * H + h][50:462, 50:462, 0]
                elif h == H - 1:
                    if w == -1:
                        merged_imgs[
                            z,
                            h * 412 : original_image_shape[1],
                            0 : original_image_shape[2],
                            0,
                        ] = imgs[z * H * W + 0 * H + h][
                            50 : original_image_shape[1] - 412 * H - 50,
                            50 : original_image_shape[2] + 50,
                            0,
                        ]
                    elif w == 0:
                        merged_imgs[
                            z, h * 412 : original_image_shape[1], 0:412, 0
                        ] = imgs[z * H * W + w * H + h][
                            50 : original_image_shape[1] - 412 * H - 50,
                            50:462,
                            0,
                        ]
                    elif w == W - 1:
                        merged_imgs[
                            z,
                            h * 412 : original_image_shape[1],
                            w * 412 : original_image_shape[2],
                            0,
                        ] = imgs[z * H * W + w * H + h][
                            50 : original_image_shape[1] - 412 * H - 50,
                            50 : original_image_shape[2] - 412 * W - 50,
                            0,
                        ]
                    else:
                        merged_imgs[
                            z,
                            h * 412 : original_image_shape[1],
                            w * 412 : (w + 1) * 412,
                            0,
                        ] = imgs[z * H * W + w * H + h][
                            50 : original_image_shape[1] - 412 * H - 50,
                            50:462,
                            0,
                        ]
                else:
                    if w == -1:
                        merged_imgs[
                            z,
                            h * 412 : (h + 1) * 412,
                            0 : original_image_shape[2],
                            0,
                        ] = imgs[z * H * W + 0 * H + h][
                            50:462, 50 : original_image_shape[2] + 50, 0
                        ]
                    elif w == 0:
                        merged_imgs[
                            z, h * 412 : (h + 1) * 412, 0:412, 0
                        ] = imgs[z * H * W + w * H + h][50:462, 50:462, 0]
                    elif w == W - 1:
                        merged_imgs[
                            z,
                            h * 412 : (h + 1) * 412,
                            w * 412 : original_image_shape[2],
                            0,
                        ] = imgs[z * H * W + w * H + h][
                            50:462,
                            50 : original_image_shape[2] - 412 * W - 50,
                            0,
                        ]
                    else:
                        merged_imgs[
                            z,
                            h * 412 : (h + 1) * 412,
                            w * 412 : (w + 1) * 412,
                            0,
                        ] = imgs[z * H * W + w * H + h][50:462, 50:462, 0]

    print(merged_imgs.shape)
    return merged_imgs


def download_model(modelname):
    """
    Downloads a specific pretained model.
    This code is adapted from DeepLabCut with permission from MWMathis
    """
    import json
    import tarfile
    import urllib.request

    def show_progress(count, block_size, total_size):
        pbar.update(block_size)

    cellseg3d_path = os.path.split(
        importlib.util.find_spec("napari_cellseg3d").origin
    )[0]
    pretrained_folder_path = os.path.join(
        cellseg3d_path, "models", "pretrained"
    )
    json_path = os.path.join(
        pretrained_folder_path, "pretrained_model_urls.json"
    )
    with open(json_path) as f:
        neturls = json.load(f)
    if modelname in neturls.keys():
        url = neturls[modelname]
        response = urllib.request.urlopen(url)
        print(
            f"Downloading the model from the M.W. Mathis Lab server {url}...."
        )
        total_size = int(response.getheader("Content-Length"))
        pbar = tqdm(unit="B", total=total_size, position=0)
        filename, _ = urllib.request.urlretrieve(url, reporthook=show_progress)
        with tarfile.open(filename, mode="r:gz") as tar:
            tar.extractall(pretrained_folder_path)
        return pretrained_folder_path
    else:
        raise ValueError(
            f"Unknown model. `modelname` should be one of {', '.join(neturls)}"
        )
