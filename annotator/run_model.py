"""
Data organization
├─ original   
        ── 0001.png   
        ── 0002.png   
        ── ...   
│   
├── labels   
        ── 0001.png   
        ── 0002.png   
        ── ...   
        └── _train0.csv
├── model_output_dir   
└── result_output_dir
"""

import matplotlib.pyplot as plt
import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, RMSprop
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2

from tqdm import tqdm
import pandas as pd


def normalize_x(image):
    return image / 127.5 - 1


def normalize_y(image):
    return image / 255


def denormalize_y(image):
    return image * 255


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
    return images, image_files


def select_train_data(dataframe, ori_imgs, label_imgs, ori_filenames, label_filenames):
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


def divide_imgs(images):
    H = -(-images.shape[1] // 412)
    W = -(-images.shape[2] // 412)

    diveded_imgs = np.zeros((images.shape[0] * H * W, 128, 128, 1), np.float32)
    print(H, W)

    for z in range(images.shape[0]):
        image = images[z]
        for h in range(H):
            for w in range(W):
                cropped_img = np.zeros((128, 128, 1), np.float32)
                cropped_img -= 1

                if images.shape[1] < 412:
                    h = -1
                if images.shape[2] < 412:
                    w = -1

                if h == -1:
                    if w == -1:
                        cropped_img[10:images.shape[1] + 10, 10:images.shape[2] + 10, 0] = image[0:images.shape[1],
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
                                                                                                  10:
                                                                                                  original_image_shape[
                                                                                                      1] + 10, 10:
                                                                                                               original_image_shape[
                                                                                                                   2] + 10,
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


def standard_uint(input_tensor, nb_filter):
    x = Conv2D(nb_filter, (3, 3), padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(nb_filter, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x


def get_nested_unet(input_shape=(128, 128, 1), num_classes=3, deep_supervision=False):
    with tf.device('/gpu:0'):

        inputs = Input(shape=input_shape)

        # 512
        conv1_1 = standard_uint(inputs, nb_filter=16)
        pool1 = MaxPooling2D((2, 2), strides=(2, 2))(conv1_1)

        # 256
        conv2_1 = standard_uint(pool1, nb_filter=32)
        pool2 = MaxPooling2D((2, 2), strides=(2, 2))(conv2_1)

        up1_2 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(conv2_1)
        conv1_2 = concatenate([up1_2, conv1_1], axis=3)
        conv1_2 = standard_uint(conv1_2, nb_filter=16)

        # 128
        conv3_1 = standard_uint(pool2, nb_filter=64)
        pool3 = MaxPooling2D((2, 2), strides=(2, 2))(conv3_1)

        up2_2 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv3_1)
        conv2_2 = concatenate([up2_2, conv2_1], axis=3)
        conv2_2 = standard_uint(conv2_2, nb_filter=32)

        up1_3 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(conv2_2)
        conv1_3 = concatenate([up1_3, conv1_1, conv1_2], axis=3)
        conv1_3 = standard_uint(conv1_3, nb_filter=16)

        # 64
        conv4_1 = standard_uint(pool3, nb_filter=128)
        pool4 = MaxPooling2D((2, 2), strides=(2, 2))(conv4_1)

        up3_2 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv4_1)
        conv3_2 = concatenate([up3_2, conv3_1], axis=3)
        conv3_2 = standard_uint(conv3_2, nb_filter=64)

        up2_3 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv3_2)
        conv2_3 = concatenate([up2_3, conv2_1, conv2_2], axis=3)
        conv2_3 = standard_uint(conv2_3, nb_filter=32)

        up1_4 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(conv2_3)
        conv1_4 = concatenate([up1_4, conv1_2, conv1_3], axis=3)
        conv1_4 = standard_uint(conv1_4, nb_filter=16)

        # 32
        conv5_1 = standard_uint(pool4, nb_filter=256)

        up4_2 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv5_1)
        conv4_2 = concatenate([up4_2, conv4_1], axis=3)
        conv4_2 = standard_uint(conv4_2, nb_filter=128)

        up3_3 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv4_2)
        conv3_3 = concatenate([up3_3, conv3_1, conv3_2], axis=3)
        conv3_3 = standard_uint(conv3_3, nb_filter=64)

        up2_4 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv3_3)
        conv2_4 = concatenate([up2_4, conv2_1, conv2_2, conv2_3], axis=3)
        conv2_4 = standard_uint(conv2_4, nb_filter=32)

        up1_5 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(conv2_4)
        conv1_5 = concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4], axis=3)
        conv1_5 = standard_uint(conv1_5, nb_filter=16)

        nested_output_1 = Conv2D(num_classes, (1, 1), activation='sigmoid')(conv1_2)
        nested_output_2 = Conv2D(num_classes, (1, 1), activation='sigmoid')(conv1_3)
        nested_output_3 = Conv2D(num_classes, (1, 1), activation='sigmoid')(conv1_4)
        nested_output_4 = Conv2D(num_classes, (1, 1), activation='sigmoid')(conv1_5)

        if deep_supervision:
            model = Model(inputs=inputs, outputs=[nested_output_1, nested_output_2, nested_output_3, nested_output_4])
        else:
            model = Model(inputs=inputs, outputs=[nested_output_4])

        model.compile(optimizer=RMSprop(lr=0.0001), loss=bce_dice_loss, metrics=[dice_coeff])

        return model


def train_unet(X_train, Y_train, csv_path, model_path, input_shape=(128, 128, 1), num_classes=1):
    Y_train = Y_train
    X_train = X_train

    data_gen_args = dict(
        rotation_range=90.,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True
    )
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 1
    image_datagen.fit(X_train, augment=True, seed=seed)
    mask_datagen.fit(Y_train, augment=True, seed=seed)

    image_generator = image_datagen.flow(X_train, seed=seed, batch_size=8)
    mask_generator = mask_datagen.flow(Y_train, seed=seed, batch_size=8)

    # combine generators into one which yields image and masks
    train_generator = (pair for pair in zip(image_generator, mask_generator))

    model = get_nested_unet(input_shape=input_shape, num_classes=num_classes)

    BATCH_SIZE = 4
    NUM_EPOCH = 100

    callbacks = []
    from tensorflow.keras.callbacks import CSVLogger
    callbacks.append(CSVLogger(csv_path))
    history = model.fit_generator(train_generator, steps_per_epoch=32, epochs=NUM_EPOCH, verbose=1, callbacks=callbacks)
    model.save_weights(model_path)


def predict(X_test, model_path, out_dir, input_shape=(512, 512, 1), num_classes=1):
    model = get_nested_unet(input_shape=input_shape, num_classes=num_classes)

    model.load_weights(model_path)
    BATCH_SIZE = 1
    Y_pred = model.predict(X_test, BATCH_SIZE)

    print(Y_pred.shape)
    os.makedirs(out_dir, exist_ok=True)

    if Y_pred.shape[3] != 1:
        num = Y_pred.shape[3]
        for n in range(num):
            os.makedirs(os.path.join(out_dir, str(n + 1)), exist_ok=True)
        for i, y in enumerate(Y_pred):
            for n in range(num):
                cv2.imwrite(os.path.join(out_dir, str(n + 1), str(i).zfill(6) + '.png'), denormalize_y(y[:, :, n]))

    else:
        for i, y in enumerate(Y_pred):
            cv2.imwrite(os.path.join(out_dir, str(i).zfill(6) + '.png'), denormalize_y(y))


def make_mask_img(ori_img, mask_img):
    mask_img_rgb = np.zeros((mask_img.shape[0], mask_img.shape[1], 3), np.float32)
    mask_img_rgb[:, :, 0] = mask_img[:, :, 0]
    mask_img_rgb[:, :, 2] = mask_img[:, :, 0]
    masked_img = cv2.addWeighted(mask_img_rgb, 0.5, cv2.cvtColor(ori_img + 0.75, cv2.COLOR_GRAY2BGR), 0.6, 0)
    return masked_img


def get_newest_csv(labelpath):
    csvs = sorted(list(Path(labelpath).glob('./*csv')))
    csv = pd.read_csv(str(csvs[-1]), index_col=0)
    return csv


"""Train"""

ori_img_dir = "/content/gdrive/MyDrive/3d/img2png"
label_dir = "/content/gdrive/MyDrive/3d/labels2png"



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
    print(img.shape)
    '''
    print(np.unique(img))
    ret , image = cv2.threshold(img , 0.1 , 1 , cv2.THRESH_BINARY)
    print(image.shape)
    print(np.unique(image))
    '''

    images = np.zeros((len(image_files), img.shape[0], img.shape[1], 1), np.float32)

    for i, image_file in tqdm(enumerate(image_files)):
        image = cv2.imread(folder_path + os.sep + image_file, cv2.IMREAD_GRAYSCALE)
        if thresh:
            ret, image = cv2.threshold(image, thresh, 1, cv2.THRESH_BINARY)
        image = image[:, :, np.newaxis]
        if normalize:
            images[i] = normalize_y(image)
        else:
            images[i] = image

    print(images.shape)

    return images, image_files


ori_imgs, ori_filenames = load_X_gray(ori_img_dir)
label_imgs, label_filenames = load_Y_gray(label_dir, thresh=0.9, normalize=False)
train_csv = get_newest_csv(label_dir)

train_ori_imgs, train_label_imgs = select_train_data(
    dataframe=train_csv,
    ori_imgs=ori_imgs,
    label_imgs=label_imgs,
    ori_filenames=ori_filenames,
    label_filenames=label_filenames
)

devided_train_ori_imgs = divide_imgs(train_ori_imgs)
devided_train_label_imgs = divide_imgs(train_label_imgs)
devided_train_label_imgs = np.where(
    devided_train_label_imgs < 0,
    0,
    devided_train_label_imgs
)

devided_train_label_imgs.shape

plt.figure(figsize=(15, 15))

plt.subplot(2, 2, 1)
plt.imshow(devided_train_ori_imgs[-1][:, :, 0], "gray")

plt.subplot(2, 2, 2)
plt.imshow(devided_train_label_imgs[-1][:, :, 0], "gray")

plt.subplot(2, 2, 3)
plt.imshow(devided_train_ori_imgs[-10][:, :, 0], "gray")

plt.subplot(2, 2, 4)
plt.imshow(devided_train_label_imgs[-10][:, :, 0], "gray")

# If you want to save trained model, specify the path on google drive for saving model, the path should look like this "/content/drive/My Drive/project/model_output_dir"
if demo:
    model_dir = "/content"
else:
    model_dir = "/content/gdrive/MyDrive/3d/modeloutput_20notchecked"

train_unet(
    X_train=devided_train_ori_imgs,
    Y_train=devided_train_label_imgs,
    csv_path=os.path.join(model_dir, "train_log.csv"),
    model_path=os.path.join(model_dir, "demo.hdf5"),
    input_shape=(128, 128, 1),
    num_classes=1
)

1.37
0.08

"""##4. predict"""

# If you want to save results, specify the directory path on google drive for saving results, the path should look like this "/content/drive/My Drive/project/result_output_dir"
if demo:
    output_dir = "/content"
else:
    output_dir = "/content/gdrive/MyDrive/3d/results3"

# XY
seped_xy_imgs = divide_imgs(ori_imgs)

predict(
    X_test=seped_xy_imgs,
    model_path=os.path.join(model_dir, "demo.hdf5"),
    out_dir=os.path.join(output_dir, "./pred"),
    input_shape=(128, 128, 1),
    num_classes=1
)

# YZ
seped_yz_imgs = divide_imgs(ori_imgs.transpose(2, 0, 1, 3))

predict(
    X_test=seped_yz_imgs,
    model_path=os.path.join(model_dir, "demo.hdf5"),
    out_dir=os.path.join(output_dir, "./pred_yz"),
    input_shape=(128, 128, 1),
    num_classes=1
)

# ZX
seped_zx_imgs = divide_imgs(ori_imgs.transpose(1, 2, 0, 3))

predict(
    X_test=seped_zx_imgs,
    model_path=os.path.join(model_dir, "demo.hdf5"),
    out_dir=os.path.join(output_dir, "./pred_zx"),
    input_shape=(512, 512, 1),
    num_classes=1
)

"""## 5. merge predict results"""

ori_image_shape = ori_imgs.shape

pred_xy_imgs, _ = load_Y_gray(os.path.join(output_dir, "./pred_xy"))
merged_imgs_xy = merge_imgs(pred_xy_imgs, ori_image_shape)

pred_yz_imgs, _ = load_Y_gray(os.path.join(output_dir, "./pred_yz"))
merged_imgs_yz = merge_imgs(pred_yz_imgs,
                            (ori_image_shape[2], ori_image_shape[0], ori_image_shape[1], ori_image_shape[3]))

pred_zx_imgs, _ = load_Y_gray(os.path.join(output_dir, "./pred_zx"))
merged_imgs_zx = merge_imgs(pred_zx_imgs,
                            (ori_image_shape[1], ori_image_shape[2], ori_image_shape[0], ori_image_shape[3]))

mito_imgs_ave = merged_imgs_xy * 255 // 3 + merged_imgs_yz.transpose(1, 2, 0, 3) * 255 // 3 + merged_imgs_zx.transpose(
    2, 0, 1, 3) * 255 // 3

out_dir = os.path.join(output_dir, './merged_prediction')
os.makedirs(out_dir, exist_ok=True)
os.makedirs(f"{out_dir}_raw", exist_ok=True)

for i in range(mito_imgs_ave.shape[0]):
    # threshed
    img = np.where(
        mito_imgs_ave[:, :, :, 0][i] >= 127,
        1,
        0
    )
    cv2.imwrite(f'{out_dir}/{str(i).zfill(4)}.png', img)

    # averaged
    img_ = np.where(
        mito_imgs_ave[:, :, :, 0][i] >= 127,
        mito_imgs_ave[:, :, :, 0][i],
        0
    )
    cv2.imwrite(f'{out_dir}_raw/{str(i).zfill(4)}.png', img_)

# WITHOUT TAP
out_dir = os.path.join(output_dir, './pred_3')
os.makedirs(out_dir, exist_ok=True)
os.makedirs(f"{out_dir}_raw", exist_ok=True)

pred_xy_imgs, _ = load_Y_gray(os.path.join(output_dir, "./pred"))
merged_imgs_xy = merge_imgs(pred_xy_imgs, ori_image_shape)
mito_imgs_ave = merged_imgs_xy  # * 255
for i in range(mito_imgs_ave.shape[0]):
    # threshed
    img = np.where(
        mito_imgs_ave[:, :, :, 0][i] >= 127,
        1,
        0
    )
    cv2.imwrite(f'{out_dir}/{str(i).zfill(4)}.png', img)

    # averaged
    img_ = np.where(
        mito_imgs_ave[:, :, :, 0][i] >= 127,
        mito_imgs_ave[:, :, :, 0][i],
        0
    )
    cv2.imwrite(f'{out_dir}_raw/{str(i).zfill(4)}.png', img_)

"""## 6. Download results and load them into PHILOW

Download the merged_prediction directory and the merged_prediction_raw directory in the directory specified as the output destination for the results, and place them in the same directory on your local machine.    
The former is the label of the prediction result itself, and the latter is used to indicate the location of low confidence.    
It is desirable to have the following configuration in your local machine.

project   
├─ original   
        ── 0001.png   
        ── 0002.png   
        ── ...   
│   
├── merged_prediction   
        ── 0001.png  
        ── 0002.png   
        ── ...   
│   
├── merged_prediction_raw   
        ── 0001.png   
        ── 0002.png   
        ── ...  


Once you have them loaded in PHILOW, you can either start the next iteration or start the final corrections.   

### Have fun!
"""
