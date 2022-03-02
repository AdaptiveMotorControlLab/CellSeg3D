import os

import cv2
import numpy as np

from utils import denormalize_y, divide_imgs, load_Y_gray, merge_imgs


def predict(X_test, model, out_dir):

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


def predict_3ax(ori_imgs, model, out_dir):

    os.makedirs(out_dir, exist_ok=True)

    # XY
    seped_xy_imgs = divide_imgs(ori_imgs)

    predict(
        X_test=seped_xy_imgs,
        model=model,
        out_dir=os.path.join(out_dir, "pred_xy"),
    )

    # YZ
    seped_yz_imgs = divide_imgs(ori_imgs.transpose(2, 0, 1, 3))

    predict(
        X_test=seped_yz_imgs,
        model=model,
        out_dir=os.path.join(out_dir, "pred_yz"),
    )

    # ZX
    seped_zx_imgs = divide_imgs(ori_imgs.transpose(1, 2, 0, 3))

    predict(
        X_test=seped_zx_imgs,
        model=model,
        out_dir=os.path.join(out_dir, "pred_zx"),
    )

    ori_image_shape = ori_imgs.shape

    pred_xy_imgs, _ = load_Y_gray(os.path.join(out_dir, "pred_xy"))
    merged_imgs_xy = merge_imgs(pred_xy_imgs, ori_image_shape)

    pred_yz_imgs, _ = load_Y_gray(os.path.join(out_dir, "pred_yz"))
    merged_imgs_yz = merge_imgs(pred_yz_imgs,
                                (ori_image_shape[2], ori_image_shape[0], ori_image_shape[1], ori_image_shape[3]))

    pred_zx_imgs, _ = load_Y_gray(os.path.join(out_dir, "pred_zx"))
    merged_imgs_zx = merge_imgs(pred_zx_imgs,
                                (ori_image_shape[1], ori_image_shape[2], ori_image_shape[0], ori_image_shape[3]))

    mito_imgs_ave = merged_imgs_xy * 255 // 3 + merged_imgs_yz.transpose(1, 2, 0, 3) * 255 // 3 \
                                              + merged_imgs_zx.transpose(2, 0, 1, 3) * 255 // 3

    out_dir_merge = os.path.join(out_dir, 'merged_prediction')
    os.makedirs(out_dir_merge, exist_ok=True)
    os.makedirs(f"{out_dir_merge}_raw", exist_ok=True)

    for i in range(mito_imgs_ave.shape[0]):
        # threshed
        img = np.where(
            mito_imgs_ave[:, :, :, 0][i] >= 127,
            1,
            0
        )
        cv2.imwrite(f'{out_dir_merge}/{str(i).zfill(4)}.png', img)

        # averaged
        img_ = np.where(
            mito_imgs_ave[:, :, :, 0][i] >= 127,
            mito_imgs_ave[:, :, :, 0][i],
            0
        )
        cv2.imwrite(f'{out_dir_merge}_raw/{str(i).zfill(4)}.png', img_)


def predict_1ax(ori_imgs, model, out_dir):

    os.makedirs(out_dir, exist_ok=True)

    # XY
    seped_xy_imgs = divide_imgs(ori_imgs)

    predict(
        X_test=seped_xy_imgs,
        model=model,
        out_dir=os.path.join(out_dir, "pred_xy"),
    )

    ori_image_shape = ori_imgs.shape

    pred_xy_imgs, _ = load_Y_gray(os.path.join(out_dir, "pred_xy"))
    merged_imgs_xy = merge_imgs(pred_xy_imgs, ori_image_shape)

    mito_imgs_ave = merged_imgs_xy * 255

    out_dir_merge = os.path.join(out_dir, 'merged_prediction')
    os.makedirs(out_dir_merge, exist_ok=True)
    os.makedirs(f"{out_dir_merge}_raw", exist_ok=True)

    for i in range(mito_imgs_ave.shape[0]):
        # threshed
        img = np.where(
            mito_imgs_ave[:, :, :, 0][i] >= 127,
            1,
            0
        )
        cv2.imwrite(f'{out_dir_merge}/{str(i).zfill(4)}.png', img)

        # averaged
        img_ = np.where(
            mito_imgs_ave[:, :, :, 0][i] >= 127,
            mito_imgs_ave[:, :, :, 0][i],
            0
        )
        cv2.imwrite(f'{out_dir_merge}_raw/{str(i).zfill(4)}.png', img_)