"""
This file contains the code to train the WNet model.
"""
# import napari
import glob
import time
from pathlib import Path
from typing import Union
from warnings import warn

import numpy as np
import tifffile as tiff
import torch
import torch.nn as nn

# MONAI
from monai.data import (
    CacheDataset,
    DataLoader,
    PatchDataset,
    pad_list_data_collate,
)
from monai.data.meta_obj import set_track_meta
from monai.metrics import DiceMetric
from monai.transforms import (
    AsDiscrete,
    Compose,
    EnsureChannelFirst,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    RandSpatialCropSamplesd,
    ScaleIntensityRanged,
    SpatialPadd,
    ToTensor,
)
from monai.utils.misc import set_determinism

# local
from napari_cellseg3d.code_models.models.wnet.model import WNet
from napari_cellseg3d.code_models.models.wnet.soft_Ncuts import SoftNCutsLoss
from napari_cellseg3d.utils import LOGGER as logger
from napari_cellseg3d.utils import dice_coeff, get_padding_dim

try:
    import wandb

    WANDB_INSTALLED = True
except ImportError:
    warn(
        "wandb not installed, wandb config will not be taken into account",
        stacklevel=1,
    )
    WANDB_INSTALLED = False

__author__ = "Yves Paychère, Colin Hofmann, Cyril Achard"


##########################
#     Utils functions    #
##########################


def create_dataset_dict(volume_directory, label_directory):
    """Creates data dictionary for MONAI transforms and training."""
    images_filepaths = sorted(
        [str(file) for file in Path(volume_directory).glob("*.tif")]
    )

    labels_filepaths = sorted(
        [str(file) for file in Path(label_directory).glob("*.tif")]
    )
    if len(images_filepaths) == 0 or len(labels_filepaths) == 0:
        raise ValueError(
            f"Data folders are empty \n{volume_directory} \n{label_directory}"
        )

    logger.info("Images :")
    for file in images_filepaths:
        logger.info(Path(file).stem)
    logger.info("*" * 10)
    logger.info("Labels :")
    for file in labels_filepaths:
        logger.info(Path(file).stem)
    try:
        data_dicts = [
            {"image": image_name, "label": label_name}
            for image_name, label_name in zip(
                images_filepaths, labels_filepaths
            )
        ]
    except ValueError as e:
        raise ValueError(
            f"Number of images and labels does not match : \n{volume_directory} \n{label_directory}"
        ) from e
    # print(f"Loaded eval image: {data_dicts}")
    return data_dicts


def create_dataset_dict_no_labs(volume_directory):
    """Creates unsupervised data dictionary for MONAI transforms and training."""
    images_filepaths = sorted(glob.glob(str(Path(volume_directory) / "*.tif")))
    if len(images_filepaths) == 0:
        raise ValueError(f"Data folder {volume_directory} is empty")

    logger.info("Images :")
    for file in images_filepaths:
        logger.info(Path(file).stem)
    logger.info("*" * 10)

    return [{"image": image_name} for image_name in images_filepaths]


def remap_image(
    image: Union[np.ndarray, torch.Tensor], new_max=100, new_min=0
):
    """Normalizes a numpy array or Tensor using the max and min value"""
    shape = image.shape
    image = image.flatten()
    image = (image - image.min()) / (image.max() - image.min())
    image = image * (new_max - new_min) + new_min
    # image = set_quantile_to_value(image)
    return image.reshape(shape)


################################
#        Config & WANDB        #
################################


class Config:
    def __init__(self):
        # WNet
        self.in_channels = 1
        self.out_channels = 1
        self.num_classes = 2
        self.dropout = 0.65
        self.use_clipping = False
        self.clipping = 1

        self.lr = 1e-6
        self.scheduler = "None"  # "CosineAnnealingLR"  # "ReduceLROnPlateau"
        self.weight_decay = 0.01  # None

        self.intensity_sigma = 1
        self.spatial_sigma = 4
        self.radius = 2  # yields to a radius depending on the data shape

        self.n_cuts_weight = 0.5
        self.reconstruction_loss = "MSE"  # "BCE"
        self.rec_loss_weight = 0.5 / 100

        self.num_epochs = 100
        self.val_interval = 5
        self.batch_size = 2
        self.num_workers = 4

        # CRF
        self.sa = 50  # 10
        self.sb = 20
        self.sg = 1
        self.w1 = 50  # 10
        self.w2 = 20
        self.n_iter = 5

        # Data
        self.train_volume_directory = "./../dataset/VIP_full"
        self.eval_volume_directory = "./../dataset/VIP_cropped/eval/"
        self.normalize_input = True
        self.normalizing_function = remap_image  # normalize_quantile
        self.use_patch = False
        self.patch_size = (64, 64, 64)
        self.num_patches = 30
        self.eval_num_patches = 20
        self.do_augmentation = True
        self.parallel = False

        self.save_model = True
        self.save_model_path = (
            r"./../results/new_model/wnet_new_model_all_data_3class.pth"
        )
        # self.save_losses_path = (
        #     r"./../results/new_model/wnet_new_model_all_data_3class.pkl"
        # )
        self.save_every = 5
        self.weights_path = None


c = Config()
###############
# Scheduler config
###############
schedulers = {
    "ReduceLROnPlateau": {
        "factor": 0.5,
        "patience": 50,
    },
    "CosineAnnealingLR": {
        "T_max": 25000,
        "eta_min": 1e-8,
    },
    "CosineAnnealingWarmRestarts": {
        "T_0": 50000,
        "eta_min": 1e-8,
        "T_mult": 1,
    },
    "CyclicLR": {
        "base_lr": 2e-7,
        "max_lr": 2e-4,
        "step_size_up": 250,
        "mode": "triangular",
    },
}

###############
# WANDB_CONFIG
###############
WANDB_MODE = "disabled"
# WANDB_MODE = "online"

WANDB_CONFIG = {
    # data setting
    "num_workers": c.num_workers,
    "normalize": c.normalize_input,
    "use_patch": c.use_patch,
    "patch_size": c.patch_size,
    "num_patches": c.num_patches,
    "eval_num_patches": c.eval_num_patches,
    "do_augmentation": c.do_augmentation,
    "model_save_path": c.save_model_path,
    # train setting
    "batch_size": c.batch_size,
    "learning_rate": c.lr,
    "weight_decay": c.weight_decay,
    "scheduler": {
        "name": c.scheduler,
        "ReduceLROnPlateau_config": {
            "factor": schedulers["ReduceLROnPlateau"]["factor"],
            "patience": schedulers["ReduceLROnPlateau"]["patience"],
        },
        "CosineAnnealingLR_config": {
            "T_max": schedulers["CosineAnnealingLR"]["T_max"],
            "eta_min": schedulers["CosineAnnealingLR"]["eta_min"],
        },
        "CosineAnnealingWarmRestarts_config": {
            "T_0": schedulers["CosineAnnealingWarmRestarts"]["T_0"],
            "eta_min": schedulers["CosineAnnealingWarmRestarts"]["eta_min"],
            "T_mult": schedulers["CosineAnnealingWarmRestarts"]["T_mult"],
        },
        "CyclicLR_config": {
            "base_lr": schedulers["CyclicLR"]["base_lr"],
            "max_lr": schedulers["CyclicLR"]["max_lr"],
            "step_size_up": schedulers["CyclicLR"]["step_size_up"],
            "mode": schedulers["CyclicLR"]["mode"],
        },
    },
    "max_epochs": c.num_epochs,
    "save_every": c.save_every,
    "val_interval": c.val_interval,
    # loss
    "reconstruction_loss": c.reconstruction_loss,
    "loss weights": {
        "n_cuts_weight": c.n_cuts_weight,
        "rec_loss_weight": c.rec_loss_weight,
    },
    "loss_params": {
        "intensity_sigma": c.intensity_sigma,
        "spatial_sigma": c.spatial_sigma,
        "radius": c.radius,
    },
    # model
    "model_type": "wnet",
    "model_params": {
        "in_channels": c.in_channels,
        "out_channels": c.out_channels,
        "num_classes": c.num_classes,
        "dropout": c.dropout,
        "use_clipping": c.use_clipping,
        "clipping_value": c.clipping,
    },
    # CRF
    "crf_params": {
        "sa": c.sa,
        "sb": c.sb,
        "sg": c.sg,
        "w1": c.w1,
        "w2": c.w2,
        "n_iter": c.n_iter,
    },
}


def train(weights_path=None, train_config=None):
    if train_config is None:
        config = Config()
    ##############
    # disable metadata tracking
    set_track_meta(False)
    ##############
    if WANDB_INSTALLED:
        wandb.init(
            config=WANDB_CONFIG, project="WNet-benchmark", mode=WANDB_MODE
        )

    set_determinism(seed=34936339)  # use default seed from NP_MAX
    torch.use_deterministic_algorithms(True, warn_only=True)

    config = train_config
    normalize_function = config.normalizing_function
    CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if CUDA else "cpu")

    print(f"Using device: {device}")

    print("Config:")
    [print(a) for a in config.__dict__.items()]

    print("Initializing training...")
    print("Getting the data")

    if config.use_patch:
        (data_shape, dataset) = get_patch_dataset(config)
    else:
        (data_shape, dataset) = get_dataset(config)
        transform = Compose(
            [
                ToTensor(),
                EnsureChannelFirst(channel_dim=0),
            ]
        )
        dataset = [transform(im) for im in dataset]
        for data in dataset:
            print(f"data shape: {data.shape}")
            break

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=pad_list_data_collate,
    )

    if config.eval_volume_directory is not None:
        eval_dataset = get_patch_eval_dataset(config)

        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=pad_list_data_collate,
        )

    dice_metric = DiceMetric(
        include_background=False, reduction="mean", get_not_nans=False
    )
    ###################################################
    #               Training the model                #
    ###################################################
    print("Initializing the model:")

    print("- getting the model")
    # Initialize the model
    model = WNet(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        num_classes=config.num_classes,
        dropout=config.dropout,
    )
    model = (
        nn.DataParallel(model).cuda() if CUDA and config.parallel else model
    )
    model.to(device)

    if config.use_clipping:
        for p in model.parameters():
            p.register_hook(
                lambda grad: torch.clamp(
                    grad, min=-config.clipping, max=config.clipping
                )
            )

    if WANDB_INSTALLED:
        wandb.watch(model, log_freq=100)

    if weights_path is not None:
        model.load_state_dict(torch.load(weights_path, map_location=device))

    print("- getting the optimizers")
    # Initialize the optimizers
    if config.weight_decay is not None:
        decay = config.weight_decay
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.lr, weight_decay=decay
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    print("- getting the loss functions")
    # Initialize the Ncuts loss function
    criterionE = SoftNCutsLoss(
        data_shape=data_shape,
        device=device,
        intensity_sigma=config.intensity_sigma,
        spatial_sigma=config.spatial_sigma,
        radius=config.radius,
    )

    if config.reconstruction_loss == "MSE":
        criterionW = nn.MSELoss()
    elif config.reconstruction_loss == "BCE":
        criterionW = nn.BCELoss()
    else:
        raise ValueError(
            f"Unknown reconstruction loss : {config.reconstruction_loss} not supported"
        )

    print("- getting the learning rate schedulers")
    # Initialize the learning rate schedulers
    scheduler = get_scheduler(config, optimizer)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode="min", factor=0.5, patience=10, verbose=True
    # )
    model.train()

    print("Ready")
    print("Training the model")
    print("*" * 50)

    startTime = time.time()
    ncuts_losses = []
    rec_losses = []
    total_losses = []
    best_dice = -1
    best_dice_epoch = -1

    # Train the model
    for epoch in range(config.num_epochs):
        print(f"Epoch {epoch + 1} of {config.num_epochs}")

        epoch_ncuts_loss = 0
        epoch_rec_loss = 0
        epoch_loss = 0

        for _i, batch in enumerate(dataloader):
            # raise NotImplementedError("testing")
            if config.use_patch:
                image = batch["image"].to(device)
            else:
                image = batch.to(device)
                if config.batch_size == 1:
                    image = image.unsqueeze(0)
                else:
                    image = image.unsqueeze(0)
                    image = torch.swapaxes(image, 0, 1)

            # Forward pass
            enc = model.forward_encoder(image)
            # out = model.forward(image)

            # Compute the Ncuts loss
            Ncuts = criterionE(enc, image)
            epoch_ncuts_loss += Ncuts.item()
            if WANDB_INSTALLED:
                wandb.log({"Ncuts loss": Ncuts.item()})

            # Forward pass
            enc, dec = model(image)

            # Compute the reconstruction loss
            if isinstance(criterionW, nn.MSELoss):
                reconstruction_loss = criterionW(dec, image)
            elif isinstance(criterionW, nn.BCELoss):
                reconstruction_loss = criterionW(
                    torch.sigmoid(dec),
                    remap_image(image, new_max=1),
                )

            epoch_rec_loss += reconstruction_loss.item()
            if WANDB_INSTALLED:
                wandb.log({"Reconstruction loss": reconstruction_loss.item()})

            # Backward pass for the reconstruction loss
            optimizer.zero_grad()
            alpha = config.n_cuts_weight
            beta = config.rec_loss_weight

            loss = alpha * Ncuts + beta * reconstruction_loss
            epoch_loss += loss.item()
            if WANDB_INSTALLED:
                wandb.log({"Sum of losses": loss.item()})
            loss.backward(loss)
            optimizer.step()

            if config.scheduler == "CosineAnnealingWarmRestarts":
                scheduler.step(epoch + _i / len(dataloader))
            if (
                config.scheduler == "CosineAnnealingLR"
                or config.scheduler == "CyclicLR"
            ):
                scheduler.step()

        ncuts_losses.append(epoch_ncuts_loss / len(dataloader))
        rec_losses.append(epoch_rec_loss / len(dataloader))
        total_losses.append(epoch_loss / len(dataloader))

        if WANDB_INSTALLED:
            wandb.log({"Ncuts loss_epoch": ncuts_losses[-1]})
            wandb.log({"Reconstruction loss_epoch": rec_losses[-1]})
            wandb.log({"Sum of losses_epoch": total_losses[-1]})
            # wandb.log({"epoch": epoch})
            # wandb.log({"learning_rate model": optimizerW.param_groups[0]["lr"]})
            # wandb.log({"learning_rate encoder": optimizerE.param_groups[0]["lr"]})
            wandb.log({"learning_rate model": optimizer.param_groups[0]["lr"]})

        print("Ncuts loss: ", ncuts_losses[-1])
        if epoch > 0:
            print(
                "Ncuts loss difference: ",
                ncuts_losses[-1] - ncuts_losses[-2],
            )
        print("Reconstruction loss: ", rec_losses[-1])
        if epoch > 0:
            print(
                "Reconstruction loss difference: ",
                rec_losses[-1] - rec_losses[-2],
            )
        print("Sum of losses: ", total_losses[-1])
        if epoch > 0:
            print(
                "Sum of losses difference: ",
                total_losses[-1] - total_losses[-2],
            )

        # Update the learning rate
        if config.scheduler == "ReduceLROnPlateau":
            # schedulerE.step(epoch_ncuts_loss)
            # schedulerW.step(epoch_rec_loss)
            scheduler.step(epoch_rec_loss)
        if (
            config.eval_volume_directory is not None
            and (epoch + 1) % config.val_interval == 0
        ):
            model.eval()
            print("Validating...")
            with torch.no_grad():
                for _k, val_data in enumerate(eval_dataloader):
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )

                    # normalize val_inputs across channels
                    if config.normalize_input:
                        for i in range(val_inputs.shape[0]):
                            for j in range(val_inputs.shape[1]):
                                val_inputs[i][j] = normalize_function(
                                    val_inputs[i][j]
                                )

                    val_outputs = model.forward_encoder(val_inputs)
                    val_outputs = AsDiscrete(threshold=0.5)(val_outputs)

                    # compute metric for current iteration
                    for channel in range(val_outputs.shape[1]):
                        max_dice_channel = torch.argmax(
                            torch.Tensor(
                                [
                                    dice_coeff(
                                        y_pred=val_outputs[
                                            :,
                                            channel : (channel + 1),
                                            :,
                                            :,
                                            :,
                                        ],
                                        y_true=val_labels,
                                    )
                                ]
                            )
                        )

                    dice_metric(
                        y_pred=val_outputs[
                            :,
                            max_dice_channel : (max_dice_channel + 1),
                            :,
                            :,
                            :,
                        ],
                        y=val_labels,
                    )
                    # if plot_val_input:  # only once
                    #     logged_image = val_inputs.detach().cpu().numpy()
                    #     logged_image = np.swapaxes(logged_image, 2, 4)
                    #     logged_image = logged_image[0, :, 32, :, :]
                    #     images = wandb.Image(
                    #         logged_image, caption="Validation input"
                    #     )
                    #
                    #     wandb.log({"val/input": images})
                    #     plot_val_input = False

                    # if k == 2 and (30 <= epoch <= 50 or epoch % 100 == 0):
                    #     logged_image = val_outputs.detach().cpu().numpy()
                    #     logged_image = np.swapaxes(logged_image, 2, 4)
                    #     logged_image = logged_image[
                    #         0, max_dice_channel, 32, :, :
                    #     ]
                    #     images = wandb.Image(
                    #         logged_image, caption="Validation output"
                    #     )
                    #
                    #     wandb.log({"val/output": images})
                    # dice_metric(y_pred=val_outputs[:, 2:, :,:,:], y=val_labels)
                    # dice_metric(y_pred=val_outputs[:, 1:, :, :, :], y=val_labels)

                    # import napari
                    # view = napari.Viewer()
                    # view.add_image(val_inputs.cpu().numpy(), name="input")
                    # view.add_image(val_labels.cpu().numpy(), name="label")
                    # vis_out = np.array(
                    #     [i.detach().cpu().numpy() for i in val_outputs],
                    # dtype=np.float32,
                    # )
                    # crf_out = np.array(
                    #     [i.detach().cpu().numpy() for i in crf_outputs],
                    # dtype=np.float32,
                    # )
                    # view.add_image(vis_out, name="output")
                    # view.add_image(crf_out, name="crf_output")
                    # napari.run()

                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                print("Validation Dice score: ", metric)
                if best_dice < metric < 2:
                    best_dice = metric
                    best_dice_epoch = epoch + 1
                    if config.save_model:
                        save_best_path = Path(config.save_model_path).parents[
                            0
                        ]
                        save_best_path.mkdir(parents=True, exist_ok=True)
                        save_best_name = Path(config.save_model_path).stem
                        save_path = (
                            str(save_best_path / save_best_name)
                            + "_best_metric.pth"
                        )
                        print(f"Saving new best model to {save_path}")
                        torch.save(model.state_dict(), save_path)

                if WANDB_INSTALLED:
                    # log validation dice score for each validation round
                    wandb.log({"val/dice_metric": metric})

                # reset the status for next validation round
                dice_metric.reset()

        print(
            "ETA: ",
            (time.time() - startTime)
            * (config.num_epochs / (epoch + 1) - 1)
            / 60,
            "minutes",
        )
        print("-" * 20)

        # Save the model
        if config.save_model and epoch % config.save_every == 0:
            torch.save(model.state_dict(), config.save_model_path)
            # with open(config.save_losses_path, "wb") as f:
            #     pickle.dump((ncuts_losses, rec_losses), f)

    print("Training finished")
    print(f"Best dice metric : {best_dice}")
    if WANDB_INSTALLED and config.eval_volume_directory is not None:
        wandb.log(
            {
                "best_dice_metric": best_dice,
                "best_metric_epoch": best_dice_epoch,
            }
        )
    print("*" * 50)

    # Save the model
    if config.save_model:
        print("Saving the model to: ", config.save_model_path)
        torch.save(model.state_dict(), config.save_model_path)
        # with open(config.save_losses_path, "wb") as f:
        #     pickle.dump((ncuts_losses, rec_losses), f)
    if WANDB_INSTALLED:
        model_artifact = wandb.Artifact(
            "WNet",
            type="model",
            description="WNet benchmark",
            metadata=dict(WANDB_CONFIG),
        )
        model_artifact.add_file(config.save_model_path)
        wandb.log_artifact(model_artifact)

    return ncuts_losses, rec_losses, model


def get_dataset(config):
    """Creates a Dataset from the original data using the tifffile library

    Args:
        config (Config): The configuration object

    Returns:
        (tuple): A tuple containing the shape of the data and the dataset
    """
    train_files = create_dataset_dict_no_labs(
        volume_directory=config.train_volume_directory
    )
    train_files = [d.get("image") for d in train_files]
    volumes = tiff.imread(train_files).astype(np.float32)
    volume_shape = volumes.shape

    if config.normalize_input:
        volumes = np.array(
            [
                # mad_normalization(volume)
                config.normalizing_function(volume)
                for volume in volumes
            ]
        )
        # mean = volumes.mean(axis=0)
        # std = volumes.std(axis=0)
        # volumes = (volumes - mean) / std
        # print("NORMALIZED VOLUMES")
        # print(volumes.shape)
        # [print("MIN MAX", volume.flatten().min(), volume.flatten().max()) for volume in volumes]
        # print(volumes.mean(axis=0), volumes.std(axis=0))

    dataset = CacheDataset(data=volumes)

    return (volume_shape, dataset)

    # train_files = create_dataset_dict_no_labs(
    #     volume_directory=config.train_volume_directory
    # )
    # train_files = [d.get("image") for d in train_files]
    # volumes = []
    # for file in train_files:
    #     image = tiff.imread(file).astype(np.float32)
    #     image = np.expand_dims(image, axis=0) # add channel dimension
    #     volumes.append(image)
    # # volumes = tiff.imread(train_files).astype(np.float32)
    # volume_shape = volumes[0].shape
    # # print(volume_shape)
    #
    # if config.do_augmentation:
    #     augmentation = Compose(
    #         [
    #             ScaleIntensityRange(
    #                 a_min=0,
    #                 a_max=2000,
    #                 b_min=0.0,
    #                 b_max=1.0,
    #                 clip=True,
    #             ),
    #             RandShiftIntensity(offsets=0.1, prob=0.5),
    #             RandFlip(spatial_axis=[1], prob=0.5),
    #             RandFlip(spatial_axis=[2], prob=0.5),
    #             RandRotate90(prob=0.1, max_k=3),
    #             ]
    #     )
    # else:
    #     augmentation = None
    #
    # dataset = CacheDataset(data=np.array(volumes), transform=augmentation)
    #
    # return (volume_shape, dataset)


def get_patch_dataset(config):
    """Creates a Dataset from the original data using the tifffile library

    Args:
        config (Config): The configuration object

    Returns:
        (tuple): A tuple containing the shape of the data and the dataset
    """

    train_files = create_dataset_dict_no_labs(
        volume_directory=config.train_volume_directory
    )

    patch_func = Compose(
        [
            LoadImaged(keys=["image"], image_only=True),
            EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
            RandSpatialCropSamplesd(
                keys=["image"],
                roi_size=(
                    config.patch_size
                ),  # multiply by axis_stretch_factor if anisotropy
                # max_roi_size=(120, 120, 120),
                random_size=False,
                num_samples=config.num_patches,
            ),
            Orientationd(keys=["image"], axcodes="PLI"),
            SpatialPadd(
                keys=["image"],
                spatial_size=(get_padding_dim(config.patch_size)),
            ),
            EnsureTyped(keys=["image"]),
        ]
    )

    train_transforms = Compose(
        [
            ScaleIntensityRanged(
                keys=["image"],
                a_min=0,
                a_max=2000,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            RandFlipd(keys=["image"], spatial_axis=[1], prob=0.5),
            RandFlipd(keys=["image"], spatial_axis=[2], prob=0.5),
            RandRotate90d(keys=["image"], prob=0.1, max_k=3),
            EnsureTyped(keys=["image"]),
        ]
    )

    dataset = PatchDataset(
        data=train_files,
        samples_per_image=config.num_patches,
        patch_func=patch_func,
        transform=train_transforms,
    )

    return config.patch_size, dataset


def get_patch_eval_dataset(config):
    eval_files = create_dataset_dict(
        volume_directory=config.eval_volume_directory + "/vol",
        label_directory=config.eval_volume_directory + "/lab",
    )

    patch_func = Compose(
        [
            LoadImaged(keys=["image", "label"], image_only=True),
            EnsureChannelFirstd(
                keys=["image", "label"], channel_dim="no_channel"
            ),
            # NormalizeIntensityd(keys=["image"]) if config.normalize_input else lambda x: x,
            RandSpatialCropSamplesd(
                keys=["image", "label"],
                roi_size=(
                    config.patch_size
                ),  # multiply by axis_stretch_factor if anisotropy
                # max_roi_size=(120, 120, 120),
                random_size=False,
                num_samples=config.eval_num_patches,
            ),
            Orientationd(keys=["image", "label"], axcodes="PLI"),
            SpatialPadd(
                keys=["image", "label"],
                spatial_size=(get_padding_dim(config.patch_size)),
            ),
            EnsureTyped(keys=["image", "label"]),
        ]
    )

    eval_transforms = Compose(
        [
            EnsureTyped(keys=["image", "label"]),
        ]
    )

    return PatchDataset(
        data=eval_files,
        samples_per_image=config.eval_num_patches,
        patch_func=patch_func,
        transform=eval_transforms,
    )


def get_dataset_monai(config):
    """Creates a Dataset applying some transforms/augmentation on the data using the MONAI library

    Args:
        config (Config): The configuration object

    Returns:
        (tuple): A tuple containing the shape of the data and the dataset
    """
    train_files = create_dataset_dict_no_labs(
        volume_directory=config.train_volume_directory
    )
    # print(train_files)
    # print(len(train_files))
    # print(train_files[0])
    first_volume = LoadImaged(keys=["image"])(train_files[0])
    first_volume_shape = first_volume["image"].shape

    # Transforms to be applied to each volume
    load_single_images = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="PLI"),
            SpatialPadd(
                keys=["image"],
                spatial_size=(get_padding_dim(first_volume_shape)),
            ),
            EnsureTyped(keys=["image"]),
        ]
    )

    if config.do_augmentation:
        train_transforms = Compose(
            [
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=0,
                    a_max=2000,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
                RandFlipd(keys=["image"], spatial_axis=[1], prob=0.5),
                RandFlipd(keys=["image"], spatial_axis=[2], prob=0.5),
                RandRotate90d(keys=["image"], prob=0.1, max_k=3),
                EnsureTyped(keys=["image"]),
            ]
        )
    else:
        train_transforms = EnsureTyped(keys=["image"])

    # Create the dataset
    dataset = CacheDataset(
        data=train_files,
        transform=Compose(load_single_images, train_transforms),
    )

    return first_volume_shape, dataset


def get_scheduler(config, optimizer, verbose=False):
    scheduler_name = config.scheduler
    if scheduler_name == "None":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=100,
            eta_min=config.lr - 1e-6,
            verbose=verbose,
        )

    elif scheduler_name == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=schedulers["ReduceLROnPlateau"]["factor"],
            patience=schedulers["ReduceLROnPlateau"]["patience"],
            verbose=verbose,
        )
    elif scheduler_name == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=schedulers["CosineAnnealingLR"]["T_max"],
            eta_min=schedulers["CosineAnnealingLR"]["eta_min"],
            verbose=verbose,
        )
    elif scheduler_name == "CosineAnnealingWarmRestarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=schedulers["CosineAnnealingWarmRestarts"]["T_0"],
            eta_min=schedulers["CosineAnnealingWarmRestarts"]["eta_min"],
            T_mult=schedulers["CosineAnnealingWarmRestarts"]["T_mult"],
            verbose=verbose,
        )
    elif scheduler_name == "CyclicLR":
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=schedulers["CyclicLR"]["base_lr"],
            max_lr=schedulers["CyclicLR"]["max_lr"],
            step_size_up=schedulers["CyclicLR"]["step_size_up"],
            mode=schedulers["CyclicLR"]["mode"],
            cycle_momentum=False,
        )
    else:
        raise ValueError(f"Scheduler {scheduler_name} not provided")
    return scheduler


if __name__ == "__main__":
    weights_location = str(
        # Path(__file__).resolve().parent / "../weights/wnet.pth"
        # "../wnet_SUM_MSE_DAPI_rad2_best_metric.pth"
    )
    train(
        # weights_location
    )
