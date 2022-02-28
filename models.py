import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop

from utils import bce_dice_loss, dice_coeff


def standard_uint(input_tensor, nb_filter):
    x = Conv2D(nb_filter, (3, 3), padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(nb_filter, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x


def get_nested_unet(input_shape=(512, 512, 1), num_classes=3, deep_supervision=False):
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
