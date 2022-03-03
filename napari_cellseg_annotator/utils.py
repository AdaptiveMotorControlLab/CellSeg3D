import datetime
import os

import cv2
import dask_image.imread
import numpy as np
import pandas as pd
from tqdm import tqdm
from qtpy.QtWidgets import QWidget, QHBoxLayout
from pathlib import Path
from skimage import io
from skimage.filters import gaussian
import keras.backend as K
from keras.losses import binary_crossentropy

from qtpy.QtGui import QDesktopServices
from qtpy.QtCore import QUrl


def open_url(url):
    QDesktopServices.openUrl(
        QUrl(url, QUrl.TolerantMode))

def normalize_x(image):
    image = image / 127.5 - 1
    return image


def normalize_y(image):
    image = image / 255
    return image


def denormalize_y(image):
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
    anno = np.concatenate([mito_anno[:, :, np.newaxis], er_anno[:, :, np.newaxis]], 2)
    anno = normalize_x(anno[np.newaxis, :, :, :])
    return anno


def check_csv(project_path, ext):
    if not os.path.isfile(os.path.join(project_path, os.path.basename(project_path) + '.csv')):
        cols = ['project', 'type', 'ext', 'z', 'y', 'x', 'z_size', 'y_size', 'x_size', 'created_date', 'update_date',
                'path', 'notes']
        df = pd.DataFrame(index=[], columns=cols)
        filename_pattern_original = os.path.join(project_path, f'dataset/Original_size/Original/*{ext}')
        images_original = dask_image.imread.imread(filename_pattern_original)
        z, y, x = images_original.shape
        record = pd.Series(
            [os.path.basename(project_path), 'dataset', '.tif', 0, 0, 0, z, y, x, datetime.datetime.now(),
             '', os.path.join(project_path, 'dataset/Original_size/Original'), ''], index=df.columns)
        df = df.append(record, ignore_index=True)
        df.to_csv(os.path.join(project_path, os.path.basename(project_path) + '.csv'))
    else:
        pass


def check_annotations_dir(project_path):
    if not os.path.isdir(os.path.join(project_path, 'annotations/Original_size/master')):
        os.makedirs(os.path.join(project_path, 'annotations/Original_size/master'))
    else:
        pass


def check_zarr(project_path, ext):
    if not len(list((Path(project_path) / 'dataset' / 'Original_size').glob('./*.zarr'))):
        filename_pattern_original = os.path.join(project_path, f'dataset/Original_size/Original/*{ext}')
        images_original = dask_image.imread.imread(filename_pattern_original)
        images_original.to_zarr(os.path.join(project_path, f'dataset/Original_size/Original.zarr'))
    else:
        pass


def check(project_path, ext):
    check_csv(project_path, ext)
    check_zarr(project_path, ext)
    check_annotations_dir(project_path)


def load_images(directory):
    filename_pattern_original = os.path.join(directory)
    images_original = dask_image.imread.imread(filename_pattern_original+'/images.tif')
    return images_original


def load_predicted_masks(mito_mask_dir, er_mask_dir):
    filename_pattern_mito_label = os.path.join(mito_mask_dir, '*png')
    filename_pattern_er_label = os.path.join(er_mask_dir, '*png')
    images_mito_label = dask_image.imread.imread(filename_pattern_mito_label)
    images_mito_label = images_mito_label.compute()
    images_er_label = dask_image.imread.imread(filename_pattern_er_label)
    images_er_label = images_er_label.compute()
    base_label = (images_mito_label > 127) * 1 + (images_er_label > 127) * 2
    return base_label


def load_saved_masks(mod_mask_dir):
    filename_pattern_label = os.path.join(mod_mask_dir)
    images_label = dask_image.imread.imread(filename_pattern_label+'/images.tif')
    images_label = images_label.compute()
    base_label = images_label
    return base_label


def load_raw_masks(raw_mask_dir):
    filename_pattern_raw = os.path.join(raw_mask_dir)
    images_raw = dask_image.imread.imread(filename_pattern_raw+'/images.tif')
    images_raw = images_raw.compute()
    base_label = np.where((126 < images_raw) & (images_raw < 171), 255, 0)
    return base_label


def combine_blocks(block1, block2):
    temp_widget = QWidget()
    temp_layout = QHBoxLayout()
    temp_layout.addWidget(block1)
    temp_layout.addWidget(block2)
    temp_widget.setLayout(temp_layout)
    return temp_widget


def save_masks(labels, out_path):
    num = labels.shape[0]
    os.makedirs(out_path, exist_ok=True)
    for i in range(num):
        label = labels[i]
        io.imsave(os.path.join(out_path, str(i).zfill(4) + '.png'), label)


def load_X_gray(folder_path):
    image_files = []
    for file in os.listdir(folder_path):
        base, ext = os.path.splitext(file)
        if ext == '.png':
            image_files.append(file)
        else:
            pass

    image_files.sort()

    img = cv2.imread(folder_path + os.sep + image_files[0], cv2.IMREAD_GRAYSCALE)

    images = np.zeros((len(image_files), img.shape[0], img.shape[1], 1), np.float32)
    for i, image_file in tqdm(enumerate(image_files)):
        image = cv2.imread(folder_path + os.sep + image_file, cv2.IMREAD_GRAYSCALE)
        image = image[:, :, np.newaxis]
        images[i] = normalize_x(image)

    print(images.shape)

    return images, image_files


def load_Y_gray(folder_path, thresh=None, normalize=False):
    image_files = []
    for file in os.listdir(folder_path):
        base, ext = os.path.splitext(file)
        if ext == '.png':
            image_files.append(file)
        else:
            pass

    image_files.sort()

    img = cv2.imread(folder_path + os.sep + image_files[0], cv2.IMREAD_GRAYSCALE)

    images = np.zeros((len(image_files), img.shape[0], img.shape[1], 1), np.float32)

    for i, image_file in tqdm(enumerate(image_files)):
        image = cv2.imread(folder_path + os.sep + image_file, cv2.IMREAD_GRAYSCALE)
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
    for ori_img, label_img, train_filename in zip(ori_imgs, label_imgs, ori_filenames):
        if train_filename in train_img_names:
            train_ori_imgs.append(ori_img)
            train_label_imgs.append(label_img)

    return np.array(train_ori_imgs), np.array(train_label_imgs)


def dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score


def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss


def bce_dice_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss


def divide_imgs(images):
    H = -(-images.shape[1] // 412)
    W = -(-images.shape[2] // 412)

    diveded_imgs = np.zeros((images.shape[0] * H * W, 512, 512, 1), np.float32)
    print(H, W)

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
                        cropped_img[50:images.shape[1] + 50, 50:images.shape[2] + 50, 0] = image[0:images.shape[1],
                                                                                           0:images.shape[2], 0]
                    elif w == 0:
                        cropped_img[50:images.shape[1] + 50, 50:512, 0] = image[0:images.shape[1], 0:462, 0]
                    elif w == W - 1:
                        cropped_img[50:images.shape[1] + 50, 0:images.shape[2] - 412 * W - 50, 0] = image[
                                                                                                    0:images.shape[1],
                                                                                                    w * 412 - 50:
                                                                                                    images.shape[2], 0]
                    else:
                        cropped_img[50:images.shape[1] + 50, :, 0] = image[0:images.shape[1],
                                                                     w * 412 - 50:(w + 1) * 412 + 50, 0]
                elif h == 0:
                    if w == -1:
                        cropped_img[50:512, 50:images.shape[2] + 50, 0] = image[0:462, 0:images.shape[2], 0]
                    elif w == 0:
                        cropped_img[50:512, 50:512, 0] = image[0:462, 0:462, 0]
                    elif w == W - 1:
                        cropped_img[50:512, 0:images.shape[2] - 412 * W - 50, 0] = image[0:462,
                                                                                   w * 412 - 50:images.shape[2], 0]
                    else:
                        # cropped_img[50:512, :, 0] = image[0:462, w*412-50:(w+1)*412+50, 0]
                        try:
                            cropped_img[50:512, :, 0] = image[0:462, w * 412 - 50:(w + 1) * 412 + 50, 0]
                        except:
                            cropped_img[50:512, 0:images.shape[2] - 412 * (W - 1) - 50, 0] = image[0:462, w * 412 - 50:(
                                                                                                                                   w + 1) * 412 + 50,
                                                                                             0]
                elif h == H - 1:
                    if w == -1:
                        cropped_img[0:images.shape[1] - 412 * H - 50, 50:images.shape[2] + 50, 0] = image[h * 412 - 50:
                                                                                                          images.shape[
                                                                                                              1],
                                                                                                    0:images.shape[2],
                                                                                                    0]
                    elif w == 0:
                        cropped_img[0:images.shape[1] - 412 * H - 50, 50:512, 0] = image[h * 412 - 50:images.shape[1],
                                                                                   0:462, 0]
                    elif w == W - 1:
                        cropped_img[0:images.shape[1] - 412 * H - 50, 0:images.shape[2] - 412 * W - 50, 0] = image[
                                                                                                             h * 412 - 50:
                                                                                                             images.shape[
                                                                                                                 1],
                                                                                                             w * 412 - 50:
                                                                                                             images.shape[
                                                                                                                 2], 0]
                    else:
                        try:
                            cropped_img[0:images.shape[1] - 412 * H - 50, :, 0] = image[h * 412 - 50:images.shape[1],
                                                                                  w * 412 - 50:(w + 1) * 412 + 50, 0]
                        except:
                            cropped_img[0:images.shape[1] - 412 * H - 50, 0:images.shape[2] - 412 * (W - 1) - 50,
                            0] = image[h * 412 - 50:images.shape[1], w * 412 - 50:(w + 1) * 412 + 50, 0]
                else:
                    if w == -1:
                        cropped_img[:, 50:images.shape[2] + 50, 0] = image[h * 412 - 50:(h + 1) * 412 + 50,
                                                                     0:images.shape[2], 0]
                    elif w == 0:
                        # cropped_img[:, 50:512, 0] = image[h*412-50:(h+1)*412+50, 0:462, 0]
                        try:
                            cropped_img[:, 50:512, 0] = image[h * 412 - 50:(h + 1) * 412 + 50, 0:462, 0]
                        except:
                            cropped_img[0:images.shape[1] - 412 * H - 50 + 412, 50:512, 0] = image[h * 412 - 50:(
                                                                                                                            h + 1) * 412 + 50,
                                                                                             0:462, 0]
                    elif w == W - 1:
                        # cropped_img[:, 0:images.shape[2]-412*W-50, 0] = image[h*412-50:(h+1)*412+50, w*412-50:images.shape[2], 0]
                        try:
                            cropped_img[:, 0:images.shape[2] - 412 * W - 50, 0] = image[h * 412 - 50:(h + 1) * 412 + 50,
                                                                                  w * 412 - 50:images.shape[2], 0]
                        except:
                            cropped_img[0:images.shape[1] - 412 * H - 50 + 412, 0:images.shape[2] - 412 * W - 50,
                            0] = image[h * 412 - 50:(h + 1) * 412 + 50, w * 412 - 50:images.shape[2], 0]
                    else:
                        # cropped_img[:, :, 0] = image[h*412-50:(h+1)*412+50, w*412-50:(w+1)*412+50, 0]
                        try:
                            cropped_img[:, :, 0] = image[h * 412 - 50:(h + 1) * 412 + 50,
                                                   w * 412 - 50:(w + 1) * 412 + 50, 0]
                        except:
                            try:
                                cropped_img[:, 0:images.shape[2] - 412 * (W - 1) - 50, 0] = image[h * 412 - 50:(
                                                                                                                           h + 1) * 412 + 50,
                                                                                            w * 412 - 50:(
                                                                                                                     w + 1) * 412 + 50,
                                                                                            0]
                            except:
                                cropped_img[0:images.shape[1] - 412 * (H - 1) - 50, :, 0] = image[h * 412 - 50:(
                                                                                                                           h + 1) * 412 + 50,
                                                                                            w * 412 - 50:(
                                                                                                                     w + 1) * 412 + 50,
                                                                                            0]
                h = max(0, h)
                w = max(0, w)
                diveded_imgs[z * H * W + w * H + h] = cropped_img
                # print(z*H*W+ w*H+h)

    return diveded_imgs


def merge_imgs(imgs, original_image_shape):
    merged_imgs = np.zeros((original_image_shape[0], original_image_shape[1], original_image_shape[2], 1), np.float32)
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
                        merged_imgs[z, 0:original_image_shape[1], 0:original_image_shape[2], 0] = imgs[
                                                                                                      z * H * W + 0 * H + 0][
                                                                                                  50:
                                                                                                  original_image_shape[
                                                                                                      1] + 50, 50:
                                                                                                               original_image_shape[
                                                                                                                   2] + 50,
                                                                                                  0]
                    elif w == 0:
                        merged_imgs[z, 0:original_image_shape[1], 0:412, 0] = imgs[z * H * W + w * H + 0][
                                                                              50:original_image_shape[1] + 50, 50:462,
                                                                              0]
                    elif w == W - 1:
                        merged_imgs[z, 0:original_image_shape[1], w * 412:original_image_shape[2], 0] = imgs[
                                                                                                            z * H * W + w * H + 0][
                                                                                                        50:
                                                                                                        original_image_shape[
                                                                                                            1] + 50, 50:
                                                                                                                     original_image_shape[
                                                                                                                         2] - 412 * W - 50,
                                                                                                        0]
                    else:
                        merged_imgs[z, 0:original_image_shape[1], w * 412:(w + 1) * 412, 0] = imgs[
                                                                                                  z * H * W + w * H + 0][
                                                                                              50:original_image_shape[
                                                                                                     1] + 50, 50:462, 0]
                elif h == 0:
                    if w == -1:
                        merged_imgs[z, 0:412, 0:original_image_shape[2], 0] = imgs[z * H * W + 0 * H + h][50:462,
                                                                              50:original_image_shape[2] + 50, 0]
                    elif w == 0:
                        merged_imgs[z, 0:412, 0:412, 0] = imgs[z * H * W + w * H + h][50:462, 50:462, 0]
                    elif w == W - 1:
                        merged_imgs[z, 0:412, w * 412:original_image_shape[2], 0] = imgs[z * H * W + w * H + h][50:462,
                                                                                    50:original_image_shape[
                                                                                           2] - 412 * W - 50, 0]
                    else:
                        merged_imgs[z, 0:412, w * 412:(w + 1) * 412, 0] = imgs[z * H * W + w * H + h][50:462, 50:462, 0]
                elif h == H - 1:
                    if w == -1:
                        merged_imgs[z, h * 412:original_image_shape[1], 0:original_image_shape[2], 0] = imgs[
                                                                                                            z * H * W + 0 * H + h][
                                                                                                        50:
                                                                                                        original_image_shape[
                                                                                                            1] - 412 * H - 50,
                                                                                                        50:
                                                                                                        original_image_shape[
                                                                                                            2] + 50, 0]
                    elif w == 0:
                        merged_imgs[z, h * 412:original_image_shape[1], 0:412, 0] = imgs[z * H * W + w * H + h][
                                                                                    50:original_image_shape[
                                                                                           1] - 412 * H - 50, 50:462, 0]
                    elif w == W - 1:
                        merged_imgs[z, h * 412:original_image_shape[1], w * 412:original_image_shape[2], 0] = imgs[
                                                                                                                  z * H * W + w * H + h][
                                                                                                              50:
                                                                                                              original_image_shape[
                                                                                                                  1] - 412 * H - 50,
                                                                                                              50:
                                                                                                              original_image_shape[
                                                                                                                  2] - 412 * W - 50,
                                                                                                              0]
                    else:
                        merged_imgs[z, h * 412:original_image_shape[1], w * 412:(w + 1) * 412, 0] = imgs[
                                                                                                        z * H * W + w * H + h][
                                                                                                    50:
                                                                                                    original_image_shape[
                                                                                                        1] - 412 * H - 50,
                                                                                                    50:462, 0]
                else:
                    if w == -1:
                        merged_imgs[z, h * 412:(h + 1) * 412, 0:original_image_shape[2], 0] = imgs[
                                                                                                  z * H * W + 0 * H + h][
                                                                                              50:462,
                                                                                              50:original_image_shape[
                                                                                                     2] + 50, 0]
                    elif w == 0:
                        merged_imgs[z, h * 412:(h + 1) * 412, 0:412, 0] = imgs[z * H * W + w * H + h][50:462, 50:462, 0]
                    elif w == W - 1:
                        merged_imgs[z, h * 412:(h + 1) * 412, w * 412:original_image_shape[2], 0] = imgs[
                                                                                                        z * H * W + w * H + h][
                                                                                                    50:462, 50:
                                                                                                            original_image_shape[
                                                                                                                2] - 412 * W - 50,
                                                                                                    0]
                    else:
                        merged_imgs[z, h * 412:(h + 1) * 412, w * 412:(w + 1) * 412, 0] = imgs[z * H * W + w * H + h][
                                                                                          50:462, 50:462, 0]

    print(merged_imgs.shape)
    return merged_imgs