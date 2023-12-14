"""Script to run WNet training in Google Colab."""
import time
from pathlib import Path

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
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    AsDiscrete,
    Compose,
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
)
from monai.utils import set_determinism

# local
from napari_cellseg3d import config, utils
from napari_cellseg3d.code_models.models.wnet.model import WNet
from napari_cellseg3d.code_models.models.wnet.soft_Ncuts import SoftNCutsLoss
from napari_cellseg3d.code_models.worker_training import TrainingWorkerBase
from napari_cellseg3d.code_models.workers_utils import (
    PRETRAINED_WEIGHTS_DIR,
)

logger = utils.LOGGER
VERBOSE_SCHEDULER = True
logger.debug(f"PRETRAINED WEIGHT DIR LOCATION : {PRETRAINED_WEIGHTS_DIR}")

try:
    import wandb

    WANDB_INSTALLED = True
except ImportError:
    logger.warning(
        "wandb not installed, wandb config will not be taken into account",
        stacklevel=1,
    )
    WANDB_INSTALLED = False

# TODO subclass to reduce code duplication


class WNetTrainingWorkerColab(TrainingWorkerBase):
    """A custom worker to run WNet (unsupervised) training jobs in.

    Inherits from :py:class:`napari.qt.threading.GeneratorWorker` via :py:class:`TrainingWorkerBase`.
    """

    def __init__(
        self,
        worker_config: config.WNetTrainingWorkerConfig,
        wandb_config: config.WandBConfig = None,
    ):
        """Create a WNet training worker for Google Colab.

        Args:
            worker_config: worker configuration
            wandb_config: optional wandb configuration
        """
        super().__init__()
        self.config = worker_config
        self.wandb_config = (
            wandb_config if wandb_config is not None else config.WandBConfig()
        )

        self.dice_metric = DiceMetric(
            include_background=False, reduction="mean", get_not_nans=False
        )
        self.normalize_function = utils.remap_image
        self.start_time = time.time()
        self.ncuts_losses = []
        self.rec_losses = []
        self.total_losses = []
        self.best_dice = -1
        self.dice_values = []

        self.dataloader: DataLoader = None
        self.eval_dataloader: DataLoader = None
        self.data_shape = None

    def log(self, text):
        """Log a message to the logger and to wandb if installed."""
        logger.info(text)

    def get_patch_dataset(self, train_transforms):
        """Creates a Dataset from the original data using the tifffile library.

        Args:
            train_transforms (Compose): The transforms to apply to the data

        Returns:
            (tuple): A tuple containing the shape of the data and the dataset
        """
        patch_func = Compose(
            [
                LoadImaged(keys=["image"], image_only=True),
                EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
                RandSpatialCropSamplesd(
                    keys=["image"],
                    roi_size=(
                        self.config.sample_size
                    ),  # multiply by axis_stretch_factor if anisotropy
                    # max_roi_size=(120, 120, 120),
                    random_size=False,
                    num_samples=self.config.num_samples,
                ),
                Orientationd(keys=["image"], axcodes="PLI"),
                SpatialPadd(
                    keys=["image"],
                    spatial_size=(
                        utils.get_padding_dim(self.config.sample_size)
                    ),
                ),
                EnsureTyped(keys=["image"]),
            ]
        )
        dataset = PatchDataset(
            data=self.config.train_data_dict,
            samples_per_image=self.config.num_samples,
            patch_func=patch_func,
            transform=train_transforms,
        )

        return self.config.sample_size, dataset

    def get_dataset_eval(self, eval_dataset_dict):
        """Creates a Dataset applying some transforms/augmentation on the data using the MONAI library."""
        eval_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(
                    keys=["image", "label"], channel_dim="no_channel"
                ),
                # RandSpatialCropSamplesd(
                #     keys=["image", "label"],
                #     roi_size=(
                #         self.config.sample_size
                #     ),  # multiply by axis_stretch_factor if anisotropy
                #     # max_roi_size=(120, 120, 120),
                #     random_size=False,
                #     num_samples=self.config.num_samples,
                # ),
                Orientationd(keys=["image", "label"], axcodes="PLI"),
                # SpatialPadd(
                #     keys=["image", "label"],
                #     spatial_size=(
                #         utils.get_padding_dim(self.config.sample_size)
                #     ),
                # ),
                EnsureTyped(keys=["image", "label"]),
            ]
        )

        return CacheDataset(
            data=eval_dataset_dict,
            transform=eval_transforms,
        )

    def get_dataset(self, train_transforms):
        """Creates a Dataset applying some transforms/augmentation on the data using the MONAI library.

        Args:
            train_transforms (Compose): The transforms to apply to the data

        Returns:
            (tuple): A tuple containing the shape of the data and the dataset
        """
        train_files = self.config.train_data_dict

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
                    spatial_size=(utils.get_padding_dim(first_volume_shape)),
                ),
                EnsureTyped(keys=["image"]),
                # RemapTensord(keys=["image"], new_min=0.0, new_max=100.0),
            ]
        )

        # Create the dataset
        dataset = CacheDataset(
            data=train_files,
            transform=Compose([load_single_images, train_transforms]),
        )

        return first_volume_shape, dataset

    def _get_data(self):
        if self.config.do_augmentation:
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

        if self.config.sampling:
            logger.debug("Loading patch dataset")
            (self.data_shape, dataset) = self.get_patch_dataset(
                train_transforms
            )
        else:
            logger.debug("Loading volume dataset")
            (self.data_shape, dataset) = self.get_dataset(train_transforms)

        logger.debug(f"Data shape : {self.data_shape}")
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=pad_list_data_collate,
        )

        if self.config.eval_volume_dict is not None:
            eval_dataset = self.get_dataset_eval(self.config.eval_volume_dict)

            self.eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                collate_fn=pad_list_data_collate,
            )
        else:
            self.eval_dataloader = None
        return self.dataloader, self.eval_dataloader, self.data_shape

    def log_parameters(self):
        """Log the parameters of the training."""
        self.log("*" * 20)
        self.log("-- Parameters --")
        self.log(f"Device: {self.config.device}")
        self.log(f"Batch size: {self.config.batch_size}")
        self.log(f"Epochs: {self.config.max_epochs}")
        self.log(f"Learning rate: {self.config.learning_rate}")
        self.log(f"Validation interval: {self.config.validation_interval}")
        if self.config.weights_info.use_custom:
            self.log(f"Custom weights: {self.config.weights_info.path}")
        elif self.config.weights_info.use_pretrained:
            self.log(f"Pretrained weights: {self.config.weights_info.path}")
        if self.config.sampling:
            self.log(
                f"Using {self.config.num_samples} samples of size {self.config.sample_size}"
            )
        if self.config.do_augmentation:
            self.log("Using data augmentation")
        ##############
        self.log("-- Model --")
        self.log(f"Using {self.config.num_classes} classes")
        self.log(f"Weight decay: {self.config.weight_decay}")
        self.log("* NCuts : ")
        self.log(f"- Intensity sigma {self.config.intensity_sigma}")
        self.log(f"- Spatial sigma {self.config.spatial_sigma}")
        self.log(f"- Radius : {self.config.radius}")
        self.log(f"* Reconstruction loss : {self.config.reconstruction_loss}")
        self.log(
            f"Weighted sum : {self.config.n_cuts_weight}*NCuts + {self.config.rec_loss_weight}*Reconstruction"
        )
        ##############
        self.log("-- Data --")
        self.log("Training data :\n")
        [
            self.log(f"{v}")
            for d in self.config.train_data_dict
            for k, v in d.items()
        ]
        if self.config.eval_volume_dict is not None:
            self.log("\nValidation data :\n")
            [
                self.log(f"{k}: {v}")
                for d in self.config.eval_volume_dict
                for k, v in d.items()
            ]
        self.log("*" * 20)

    def train(
        self, provided_model=None, provided_optimizer=None, provided_loss=None
    ):
        """Train the model."""
        try:
            if self.config is None:
                self.config = config.WNetTrainingWorkerConfig()
            ##############
            # disable metadata tracking in MONAI
            set_track_meta(False)
            ##############
            if WANDB_INSTALLED:
                config_dict = self.config.__dict__
                logger.debug(f"wandb config : {config_dict}")
                wandb.init(
                    config=config_dict,
                    project="CellSeg3D (Colab)",
                    mode=self.wandb_config.mode,
                )

            set_determinism(seed=self.config.deterministic_config.seed)
            torch.use_deterministic_algorithms(True, warn_only=True)

            device = self.config.device

            self.log_parameters()
            self.log("Initializing training...")
            self.log("- Getting the data")

            self._get_data()

            ###################################################
            #               Training the model                #
            ###################################################
            self.log("- Getting the model")
            # Initialize the model
            model = (
                WNet(
                    in_channels=self.config.in_channels,
                    out_channels=self.config.out_channels,
                    num_classes=self.config.num_classes,
                    dropout=self.config.dropout,
                )
                if provided_model is None
                else provided_model
            )
            model.to(device)

            if self.config.use_clipping:
                for p in model.parameters():
                    p.register_hook(
                        lambda grad: torch.clamp(
                            grad,
                            min=-self.config.clipping,
                            max=self.config.clipping,
                        )
                    )

            if WANDB_INSTALLED:
                wandb.watch(model, log_freq=100)

            if self.config.weights_info.use_custom:
                if self.config.weights_info.use_pretrained:
                    weights_file = "wnet.pth"
                    self.downloader.download_weights("WNet", weights_file)
                    weights = PRETRAINED_WEIGHTS_DIR / Path(weights_file)
                    self.config.weights_info.path = weights
                else:
                    weights = str(Path(self.config.weights_info.path))

                try:
                    model.load_state_dict(
                        torch.load(
                            weights,
                            map_location=self.config.device,
                        ),
                        strict=True,
                    )
                except RuntimeError as e:
                    logger.error(f"Error when loading weights : {e}")
                    logger.exception(e)
                    warn = (
                        "WARNING:\nIt'd seem that the weights were incompatible with the model,\n"
                        "the model will be trained from random weights"
                    )
                    self.log(warn)
                    self.warn(warn)
                    self._weight_error = True
            else:
                self.log("Model will be trained from scratch")
            self.log("- Getting the optimizer")
            # Initialize the optimizers
            if self.config.weight_decay is not None:
                decay = self.config.weight_decay
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=self.config.learning_rate,
                    weight_decay=decay,
                )
            else:
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=self.config.learning_rate
                )
            if provided_optimizer is not None:
                optimizer = provided_optimizer
            self.log("- Getting the loss functions")
            # Initialize the Ncuts loss function
            criterionE = SoftNCutsLoss(
                data_shape=self.data_shape,
                device=device,
                intensity_sigma=self.config.intensity_sigma,
                spatial_sigma=self.config.spatial_sigma,
                radius=self.config.radius,
            )

            if self.config.reconstruction_loss == "MSE":
                criterionW = nn.MSELoss()
            elif self.config.reconstruction_loss == "BCE":
                criterionW = nn.BCELoss()
            else:
                raise ValueError(
                    f"Unknown reconstruction loss : {self.config.reconstruction_loss} not supported"
                )

            model.train()

            self.log("Ready")
            self.log("Training the model")
            self.log("*" * 20)

            # Train the model
            for epoch in range(self.config.max_epochs):
                self.log(f"Epoch {epoch + 1} of {self.config.max_epochs}")

                epoch_ncuts_loss = 0
                epoch_rec_loss = 0
                epoch_loss = 0

                for _i, batch in enumerate(self.dataloader):
                    # raise NotImplementedError("testing")
                    image_batch = batch["image"].to(device)
                    # Normalize the image
                    for i in range(image_batch.shape[0]):
                        for j in range(image_batch.shape[1]):
                            image_batch[i, j] = self.normalize_function(
                                image_batch[i, j]
                            )

                    # Forward pass
                    enc, dec = model(image_batch)
                    # Compute the Ncuts loss
                    Ncuts = criterionE(enc, image_batch)

                    epoch_ncuts_loss += Ncuts.item()
                    if WANDB_INSTALLED:
                        wandb.log({"Train/Ncuts loss": Ncuts.item()})

                    # Compute the reconstruction loss
                    if isinstance(criterionW, nn.MSELoss):
                        reconstruction_loss = criterionW(dec, image_batch)
                    elif isinstance(criterionW, nn.BCELoss):
                        reconstruction_loss = criterionW(
                            torch.sigmoid(dec),
                            utils.remap_image(image_batch, new_max=1),
                        )

                    epoch_rec_loss += reconstruction_loss.item()
                    if WANDB_INSTALLED:
                        wandb.log(
                            {
                                "Train/Reconstruction loss": reconstruction_loss.item()
                            }
                        )

                    # Backward pass for the reconstruction loss
                    optimizer.zero_grad()
                    alpha = self.config.n_cuts_weight
                    beta = self.config.rec_loss_weight

                    loss = alpha * Ncuts + beta * reconstruction_loss
                    if provided_loss is not None:
                        loss = provided_loss
                    epoch_loss += loss.item()

                    if WANDB_INSTALLED:
                        wandb.log(
                            {"Train/Weighted sum of losses": loss.item()}
                        )

                    loss.backward(loss)
                    optimizer.step()
                    yield epoch_loss

                self.ncuts_losses.append(
                    epoch_ncuts_loss / len(self.dataloader)
                )
                self.rec_losses.append(epoch_rec_loss / len(self.dataloader))
                self.total_losses.append(epoch_loss / len(self.dataloader))

                if WANDB_INSTALLED:
                    wandb.log({"Ncuts loss for epoch": self.ncuts_losses[-1]})
                    wandb.log(
                        {"Reconstruction loss for epoch": self.rec_losses[-1]}
                    )
                    wandb.log(
                        {"Sum of losses for epoch": self.total_losses[-1]}
                    )
                    wandb.log(
                        {
                            "LR/Model learning rate": optimizer.param_groups[
                                0
                            ]["lr"]
                        }
                    )

                self.log(f"Ncuts loss: {self.ncuts_losses[-1]:.5f}")
                self.log(f"Reconstruction loss: {self.rec_losses[-1]:.5f}")
                self.log(
                    f"Weighted sum of losses: {self.total_losses[-1]:.5f}"
                )
                if epoch > 0:
                    self.log(
                        f"Ncuts loss difference: {self.ncuts_losses[-1] - self.ncuts_losses[-2]:.5f}"
                    )
                    self.log(
                        f"Reconstruction loss difference: {self.rec_losses[-1] - self.rec_losses[-2]:.5f}"
                    )
                    self.log(
                        f"Weighted sum of losses difference: {self.total_losses[-1] - self.total_losses[-2]:.5f}"
                    )

                if (
                    self.eval_dataloader is not None
                    and (epoch + 1) % self.config.validation_interval == 0
                ):
                    model.eval()
                    self.log("Validating...")
                    self.eval(model, epoch)  # validation

                eta = (
                    (time.time() - self.start_time)
                    * (self.config.max_epochs / (epoch + 1) - 1)
                    / 60
                )
                self.log(f"ETA: {eta:.1f} minutes")
                self.log("-" * 20)

                # Save the model
                if epoch % 5 == 0:
                    torch.save(
                        model.state_dict(),
                        self.config.results_path_folder + "/wnet_.pth",
                    )

            self.log("Training finished")
            if self.best_dice > -1:
                best_dice_epoch = epoch
                self.log(
                    f"Best dice metric : {self.best_dice} at epoch {best_dice_epoch}"
                )

                if WANDB_INSTALLED:
                    wandb.log(
                        {
                            "Validation/Best Dice": self.best_dice,
                            "Validation/Best Dice epoch": best_dice_epoch,
                        }
                    )

            # Save the model
            self.log(
                f"Saving the model to: {self.config.results_path_folder}/wnet.pth",
            )
            save_weights_path = self.config.results_path_folder + "/wnet.pth"
            torch.save(
                model.state_dict(),
                save_weights_path,
            )

            if WANDB_INSTALLED and self.wandb_config.save_model_artifact:
                model_artifact = wandb.Artifact(
                    "WNet",
                    type="model",
                    description="CellSeg3D WNet",
                    metadata=self.config.__dict__,
                )
                model_artifact.add_file(save_weights_path)
                wandb.log_artifact(model_artifact)

        except Exception as e:
            msg = f"Training failed with exception: {e}"
            self.log(msg)
            self.raise_error(e, msg)
            self.quit()
            raise e

    def eval(self, model, _):
        """Evaluate the model on the validation set."""
        with torch.no_grad():
            device = self.config.device
            for _k, val_data in enumerate(self.eval_dataloader):
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )

                # normalize val_inputs across channels
                for i in range(val_inputs.shape[0]):
                    for j in range(val_inputs.shape[1]):
                        val_inputs[i][j] = self.normalize_function(
                            val_inputs[i][j]
                        )
                logger.debug(f"Val inputs shape: {val_inputs.shape}")
                val_outputs = sliding_window_inference(
                    val_inputs,
                    roi_size=[64, 64, 64],
                    sw_batch_size=1,
                    predictor=model.forward_encoder,
                    overlap=0.1,
                    mode="gaussian",
                    sigma_scale=0.01,
                    progress=True,
                )
                val_decoder_outputs = sliding_window_inference(
                    val_outputs,
                    roi_size=[64, 64, 64],
                    sw_batch_size=1,
                    predictor=model.forward_decoder,
                    overlap=0.1,
                    mode="gaussian",
                    sigma_scale=0.01,
                    progress=True,
                )
                val_outputs = AsDiscrete(threshold=0.5)(val_outputs)
                logger.debug(f"Val outputs shape: {val_outputs.shape}")
                logger.debug(f"Val labels shape: {val_labels.shape}")
                logger.debug(
                    f"Val decoder outputs shape: {val_decoder_outputs.shape}"
                )

                # dices = []
                # Find in which channel the labels are (avoid background)
                # for channel in range(val_outputs.shape[1]):
                #     dices.append(
                #         utils.dice_coeff(
                #             y_pred=val_outputs[
                #                 0, channel : (channel + 1), :, :, :
                #             ],
                #             y_true=val_labels[0],
                #         )
                #     )
                # logger.debug(f"DICE COEFF: {dices}")
                # max_dice_channel = torch.argmax(
                #     torch.Tensor(dices)
                # )
                # logger.debug(
                #     f"MAX DICE CHANNEL: {max_dice_channel}"
                # )
                self.dice_metric(
                    y_pred=val_outputs,
                    # [
                    #     :,
                    #     max_dice_channel : (max_dice_channel + 1),
                    #     :,
                    #     :,
                    #     :,
                    # ],
                    y=val_labels,
                )

            # aggregate the final mean dice result
            metric = self.dice_metric.aggregate().item()
            self.dice_values.append(metric)
            self.log(f"Validation Dice score: {metric:.3f}")
            if self.best_dice < metric <= 1:
                self.best_dice = metric
                # save the best model
                save_best_path = self.config.results_path_folder
                # save_best_path.mkdir(parents=True, exist_ok=True)
                save_best_name = "wnet"
                save_path = (
                    str(Path(save_best_path) / save_best_name)
                    + "_best_metric.pth"
                )
                self.log(f"Saving new best model to {save_path}")
                torch.save(model.state_dict(), save_path)

            if WANDB_INSTALLED:
                # log validation dice score for each validation round
                wandb.log({"Validation/Dice metric": metric})

            self.dice_metric.reset()

            val_decoder_outputs = None
            del val_decoder_outputs
            val_outputs = None
            del val_outputs
            val_labels = None
            del val_labels
            val_inputs = None
            del val_inputs


def get_colab_worker(
    worker_config: config.WNetTrainingWorkerConfig,
    wandb_config: config.WandBConfig,
):
    """Train a WNet model in Google Colab.

    Args:
        worker_config (config.WNetTrainingWorkerConfig): config for the training worker
        wandb_config (config.WandBConfig): config for wandb
    """
    worker = WNetTrainingWorkerColab(worker_config)
    worker.wandb_config = wandb_config
    return worker


def create_dataset_dict_no_labs(volume_directory):
    """Creates unsupervised data dictionary for MONAI transforms and training."""
    if not volume_directory.exists():
        raise ValueError(f"Data folder {volume_directory} does not exist")
    images_filepaths = utils.get_all_matching_files(volume_directory)
    if len(images_filepaths) == 0:
        raise ValueError(f"Data folder {volume_directory} is empty")

    logger.info("Images :")
    for file in images_filepaths:
        logger.info(Path(file).stem)
    logger.info("*" * 10)
    return [{"image": str(image_name)} for image_name in images_filepaths]


def create_eval_dataset_dict(image_directory, label_directory):
    """Creates data dictionary for MONAI transforms and training.

    Returns:
        A dict with the following keys

        * "image": image
        * "label" : corresponding label
    """
    images_filepaths = utils.get_all_matching_files(image_directory)
    labels_filepaths = utils.get_all_matching_files(label_directory)

    if len(images_filepaths) == 0 or len(labels_filepaths) == 0:
        raise ValueError("Data folders are empty")

    if not Path(images_filepaths[0]).parent.exists():
        raise ValueError("Images folder does not exist")
    if not Path(labels_filepaths[0]).parent.exists():
        raise ValueError("Labels folder does not exist")

    logger.info("Images :\n")
    for file in images_filepaths:
        logger.info(Path(file).name)
    logger.info("*" * 10)
    logger.info("Labels :\n")
    for file in labels_filepaths:
        logger.info(Path(file).name)
    return [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(images_filepaths, labels_filepaths)
    ]
