from tensorflow.keras.callbacks import CSVLogger
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from models import get_nested_unet


def train_unet(X_train, Y_train, csv_path, model_path, model):
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

    BATCH_SIZE = 4
    NUM_EPOCH = 400

    callbacks = []
    callbacks.append(CSVLogger(csv_path))
    history = model.fit(train_generator,steps_per_epoch=32, epochs=NUM_EPOCH, verbose=1, callbacks=callbacks)
    model.save_weights(model_path)
