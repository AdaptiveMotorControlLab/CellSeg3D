"""Contains the workers used to train the models."""
import platform
import time
from abc import abstractmethod
from math import ceil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# MONAI
from monai.data import (
    CacheDataset,
    DataLoader,
    PatchDataset,
    decollate_batch,
    pad_list_data_collate,
)
from monai.data.meta_obj import set_track_meta
from monai.inferers import sliding_window_inference
from monai.losses import (
    DiceCELoss,
    DiceLoss,
    GeneralizedDiceLoss,
    TverskyLoss,
)
from monai.metrics import DiceMetric
from monai.transforms import (
    AsDiscrete,
    Compose,
    EnsureChannelFirstd,
    EnsureType,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    Rand3DElasticd,
    RandAffined,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    RandSpatialCropSamplesd,
    SpatialPadd,
)
from monai.utils import set_determinism

# Qt
from napari.qt.threading import GeneratorWorker

# local
from napari_cellseg3d import config, utils
from napari_cellseg3d.code_models.models.wnet.model import WNet
from napari_cellseg3d.code_models.models.wnet.soft_Ncuts import SoftNCutsLoss
from napari_cellseg3d.code_models.workers_utils import (
    PRETRAINED_WEIGHTS_DIR,
    LogSignal,
    QuantileNormalizationd,
    RemapTensor,
    Threshold,
    TrainingReport,
    WeightsDownloader,
)

logger = utils.LOGGER
try:
    import wandb
    from wandb import (
        init,
    )

    # used to check if wandb is installed, otherwise not used this way
    WANDB_INSTALLED = True
except (ImportError, ModuleNotFoundError):
    logger.info(
        "wandb not installed, wandb config will not be taken into account",
        stacklevel=1,
    )
    WANDB_INSTALLED = False

VERBOSE_SCHEDULER = True
logger.debug(f"PRETRAINED WEIGHT DIR LOCATION : {PRETRAINED_WEIGHTS_DIR}")

"""
Writing something to log messages from outside the main thread needs specific care,
Following the instructions in the guides below to have a worker with custom signals,
a custom worker function was implemented.
"""

# https://python-forum.io/thread-31349.html
# https://www.pythoncentral.io/pysidepyqt-tutorial-creating-your-own-signals-and-slots/
# https://napari-staging-site.github.io/guides/stable/threading.html


class TrainingWorkerBase(GeneratorWorker):
    """A basic worker abstract class, to run training jobs in.

    Contains the minimal common elements required for training models.
    """

    wandb_config = config.WandBConfig()

    def __init__(self):
        """Initializes the worker."""
        super().__init__(self.train)
        self._signals = LogSignal()
        self.log_signal = self._signals.log_signal
        self.warn_signal = self._signals.warn_signal
        self.error_signal = self._signals.error_signal
        self.downloader = WeightsDownloader()
        self.train_files = []
        self.val_files = []
        self.config = None

        self._weight_error = False
        ################################

    def set_download_log(self, widget):
        """Sets the log widget for the downloader to output to."""
        self.downloader.log_widget = widget

    def log(self, text):
        """Sends a Qt signal that the provided text should be logged.

        Goes in a Log object, defined in :py:mod:`napari_cellseg3d.interface`.
        Sends a signal to the main thread to log the text.
        Signal is defined in napari_cellseg3d.workers_utils.LogSignal.

        Args:
            text (str): text to logged
        """
        self.log_signal.emit(text)

    def warn(self, warning):
        """Sends a warning to main thread."""
        self.warn_signal.emit(warning)

    def raise_error(self, exception, msg):
        """Sends an error to main thread."""
        logger.error(msg, exc_info=True)
        logger.error(exception, exc_info=True)
        self.error_signal.emit(exception, msg)
        self.errored.emit(exception)
        self.quit()

    @abstractmethod
    def log_parameters(self):
        """Logs the parameters of the training."""
        raise NotImplementedError

    @abstractmethod
    def train(self):
        """Starts a training job."""
        raise NotImplementedError


class WNetTrainingWorker(TrainingWorkerBase):
    """A custom worker to run WNet (unsupervised) training jobs in.

    Inherits from :py:class:`napari.qt.threading.GeneratorWorker` via :py:class:`TrainingWorkerBase`.
    """

    # TODO : add wandb parameters

    def __init__(
        self,
        worker_config: config.WNetTrainingWorkerConfig,
    ):
        """Initializes the worker.

        Args:
            worker_config (config.WNetTrainingWorkerConfig): The configuration object
        """
        super().__init__()
        self.config = worker_config

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

    def get_patch_dataset(self, train_transforms):
        """Creates a Dataset from the original data using the tifffile library.

        Args:
            train_transforms (monai.transforms.Compose): The transforms to apply to the data

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
            train_transforms (monai.transforms.Compose): The transforms to apply to the data

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
                    # ScaleIntensityRanged(
                    #     keys=["image"],
                    #     a_min=0,
                    #     a_max=2000,
                    #     b_min=0.0,
                    #     b_max=1.0,
                    #     clip=True,
                    # ),
                    # RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
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
            logger.debug(f"Eval batch size : {self.config.eval_batch_size}")
            self.eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=self.config.eval_batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                collate_fn=pad_list_data_collate,
            )
        else:
            self.eval_dataloader = None
        return self.dataloader, self.eval_dataloader, self.data_shape

    def log_parameters(self):
        """Logs the parameters of the training."""
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
        """Main training function.

        Note : args are mainly used for testing purposes. Model is otherwise initialized in the function.

        Args:
            provided_model (WNet, optional): A model to use for training. Defaults to None.
            provided_optimizer (torch.optim.Optimizer, optional): An optimizer to use for training. Defaults to None.
            provided_loss (torch.nn.Module, optional): A loss function to use for training. Defaults to None.
        """
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
                    project="CellSeg3D",
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

                    yield TrainingReport(
                        show_plot=False,
                        weights=model.state_dict(),
                        supervised=False,
                    )

                    if self._abort_requested:
                        self.dataloader = None
                        del self.dataloader
                        self.eval_dataloader = None
                        del self.eval_dataloader
                        model = None
                        del model
                        optimizer = None
                        del optimizer
                        criterionE = None
                        del criterionE
                        criterionW = None
                        del criterionW
                        torch.cuda.empty_cache()

                self.ncuts_losses.append(
                    epoch_ncuts_loss / len(self.dataloader)
                )
                self.rec_losses.append(epoch_rec_loss / len(self.dataloader))
                self.total_losses.append(epoch_loss / len(self.dataloader))

                if self.eval_dataloader is None:
                    try:
                        enc_out = enc[0].detach().cpu().numpy()
                        dec_out = dec[0].detach().cpu().numpy()
                        image_batch = image_batch[0].detach().cpu().numpy()

                        images_dict = {
                            "Encoder output": {
                                "data": enc_out,
                                "cmap": "turbo",
                            },
                            "Encoder output (discrete)": {
                                "data": AsDiscrete(threshold=0.5)(
                                    enc_out
                                ).numpy(),
                                "cmap": "bop blue",
                            },
                            "Decoder output": {
                                "data": np.squeeze(dec_out),
                                "cmap": "gist_earth",
                            },
                            "Input image": {
                                "data": np.squeeze(image_batch),
                                "cmap": "inferno",
                            },
                        }

                        yield TrainingReport(
                            show_plot=True,
                            epoch=epoch,
                            loss_1_values={"SoftNCuts": self.ncuts_losses},
                            loss_2_values=self.rec_losses,
                            weights=model.state_dict(),
                            images_dict=images_dict,
                            supervised=False,
                        )
                    except TypeError:
                        pass

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
                    yield self.eval(model, epoch)  # validation

                    if self._abort_requested:
                        self.dataloader = None
                        del self.dataloader
                        self.eval_dataloader = None
                        del self.eval_dataloader
                        model = None
                        del model
                        optimizer = None
                        del optimizer
                        criterionE = None
                        del criterionE
                        criterionW = None
                        del criterionW
                        torch.cuda.empty_cache()

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

            dataloader = None
            del dataloader
            self.eval_dataloader = None
            del self.eval_dataloader
            model = None
            del model
            optimizer = None
            del optimizer
            criterionE = None
            del criterionE
            criterionW = None
            del criterionW
            torch.cuda.empty_cache()
        except Exception as e:
            msg = f"Training failed with exception: {e}"
            self.log(msg)
            self.raise_error(e, msg)
            self.quit()
            raise e

    def eval(self, model, epoch) -> TrainingReport:
        """Evaluates the model on the validation set.

        Args:
            model (WNet): The model to evaluate
            epoch (int): The current epoch

        Returns:
            TrainingReport: A training report containing the results of the evaluation. See :py:class:`napari_cellseg3d.workers_utils.TrainingReport`
        """
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

                max_dice_channel = utils.seek_best_dice_coeff_channel(
                    y_pred=val_outputs, y_true=val_labels
                )
                self.dice_metric(
                    y_pred=val_outputs[
                        :,
                        max_dice_channel : (max_dice_channel + 1),
                        :,
                        :,
                        :,
                    ],
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
            dec_out_val = val_decoder_outputs[0].detach().cpu().numpy().copy()
            enc_out_val = val_outputs[0].detach().cpu().numpy().copy()
            lab_out_val = val_labels[0].detach().cpu().numpy().copy()
            val_in = val_inputs[0].detach().cpu().numpy().copy()

            display_dict = {
                "Reconstruction": {
                    "data": np.squeeze(dec_out_val),
                    "cmap": "gist_earth",
                },
                "Segmentation": {
                    "data": np.squeeze(enc_out_val),
                    "cmap": "turbo",
                },
                "Inputs": {
                    "data": np.squeeze(val_in),
                    "cmap": "inferno",
                },
                "Labels": {
                    "data": np.squeeze(lab_out_val),
                    "cmap": "bop blue",
                },
            }
            val_decoder_outputs = None
            del val_decoder_outputs
            val_outputs = None
            del val_outputs
            val_labels = None
            del val_labels
            val_inputs = None
            del val_inputs

            return TrainingReport(
                epoch=epoch,
                loss_1_values={
                    "SoftNCuts": self.ncuts_losses,
                    "Dice metric": self.dice_values,
                },
                loss_2_values=self.rec_losses,
                weights=model.state_dict(),
                images_dict=display_dict,
                supervised=False,
            )


class SupervisedTrainingWorker(TrainingWorkerBase):
    """A custom worker to run supervised training jobs in.

    Inherits from :py:class:`napari.qt.threading.GeneratorWorker` via :py:class:`TrainingWorkerBase`.
    """

    labels_not_semantic = False

    def __init__(
        self,
        worker_config: config.SupervisedTrainingWorkerConfig,
    ):
        """Initializes a worker for inference with the arguments needed by the :py:func:`~train` function. Note: See :py:func:`~train`.

        Config provides the following attributes:
            * device : device to train on, cuda or cpu

            * model_dict : dict containing the model's "name" and "class"

            * weights_path : path to weights files if transfer learning is to be used

            * data_dicts : dict from :py:func:`Trainer.create_train_dataset_dict`

            * validation_percent : percentage of images to use as validation

            * max_epochs : the amout of epochs to train for

            * loss_function : the loss function to use for training

            * learning_rate : the learning rate of the optimizer

            * val_interval : the interval at which to perform validation (e.g. if 2 will validate once every 2 epochs.) Also determines frequency of saving, depending on whether the metric is better or not

            * batch_size : the batch size to use for training

            * results_path : the path to save results in

            * sampling : whether to extract patches from images or not

            * num_samples : the number of samples to extract from an image for training

            * sample_size : the size of the patches to extract when sampling

            * do_augmentation : whether to perform data augmentation or not

            * deterministic : dict with "use deterministic" : bool, whether to use deterministic training, "seed": seed for RNG

        """
        super().__init__()  # worker function is self.train in parent class
        self.config = worker_config
        #######################################
        self.loss_dict = {
            "Dice": DiceLoss(sigmoid=True),
            # "BCELoss": torch.nn.BCELoss(), # dev
            # "BCELogits": torch.nn.BCEWithLogitsLoss(),
            "Generalized Dice": GeneralizedDiceLoss(sigmoid=True),
            "DiceCE": DiceCELoss(sigmoid=True, lambda_ce=0.5),
            "Tversky": TverskyLoss(sigmoid=True),
            # "Focal loss": FocalLoss(),
            # "Dice-Focal loss": DiceFocalLoss(sigmoid=True, lambda_dice=0.5),
        }
        self.loss_function = None

    def _set_loss_from_config(self):
        try:
            self.loss_function = self.loss_dict[self.config.loss_function]
        except KeyError as e:
            self.raise_error(e, "Loss function not found, aborting job")
        return self.loss_function

    def log_parameters(self):
        """Logs the parameters of the training."""
        self.log("-" * 20)
        self.log("Parameters summary :\n")

        self.log(
            f"Percentage of dataset used for training : {self.config.training_percent * 100}%"
        )

        # self.log("-" * 10)
        self.log("Training files :\n")
        [
            self.log(f"- {Path(train_file['image']).name}\n")
            for train_file in self.train_files
        ]
        # self.log("-" * 10)
        self.log("Validation files :\n")
        [
            self.log(f"- {Path(val_file['image']).name}\n")
            for val_file in self.val_files
        ]
        # self.log("-" * 10)

        if self.config.deterministic_config.enabled:
            self.log("Deterministic training is enabled")
            self.log(f"Seed is {self.config.deterministic_config.seed}")

        self.log(f"Training for {self.config.max_epochs} epochs")
        self.log(f"Loss function is : {str(self.config.loss_function)}")
        self.log(
            f"Validation is performed every {self.config.validation_interval} epochs"
        )
        self.log(f"Batch size is {self.config.batch_size}")
        self.log(f"Learning rate is {self.config.learning_rate}")

        if self.config.sampling:
            self.log(
                f"Extracting {self.config.num_samples} patches of size {self.config.sample_size}"
            )
        else:
            self.log("Using whole images as dataset")

        if self.config.do_augmentation:
            self.log("Data augmentation is enabled")

        if self.config.weights_info.use_custom:
            self.log(f"Using weights from : {self.config.weights_info.path}")
            if self._weight_error:
                self.log(
                    ">>>>>>>>>>>>>>>>>\n"
                    "WARNING:\nChosen weights were incompatible with the model,\n"
                    "the model will be trained from random weights\n"
                    "<<<<<<<<<<<<<<<<<\n"
                )

        # self.log("\n")
        # self.log("-" * 20)

    def train(
        self,
        provided_model=None,
        provided_optimizer=None,
        provided_loss=None,
        provided_scheduler=None,
    ):
        """Trains the PyTorch model for the given number of epochs.

        Uses the selected model and data, using the chosen batch size, validation interval, loss function, and number of samples.
        Will perform validation once every :py:obj:`val_interval` and save results if the mean dice is better.

        Requires:

        * device : device to train on, cuda or cpu

        * model_dict : dict containing the model's "name" and "class"

        * weights_path : path to weights files if transfer learning is to be used

        * data_dicts : dict from :py:func:`Trainer.create_train_dataset_dict`

        * validation_percent : percentage of images to use as validation

        * max_epochs : the amount of epochs to train for

        * loss_function : the loss function to use for training

        * learning rate : the learning rate of the optimizer

        * val_interval : the interval at which to perform validation (e.g. if 2 will validate once every 2 epochs.) Also determines frequency of saving, depending on whether the metric is better or not

        * batch_size : the batch size to use for training

        * results_path : the path to save results in

        * sampling : whether to extract patches from images or not

        * num_samples : the number of samples to extract from an image for training

        * sample_size : the size of the patches to extract when sampling

        * do_augmentation : whether to perform data augmentation or not

        * deterministic : dict with "use deterministic" : bool, whether to use deterministic training, "seed": seed for RNG
        """
        #########################
        # error_log = open(results_path +"/error_log.log" % multiprocessing.current_process().name, 'x')
        # faulthandler.enable(file=error_log, all_threads=True)
        #########################
        model_config = self.config.model_info
        weights_config = self.config.weights_info
        deterministic_config = self.config.deterministic_config

        start_time = time.time()

        try:
            if WANDB_INSTALLED:
                config_dict = self.config.__dict__
                logger.debug(f"wandb config : {config_dict}")
                try:
                    wandb.init(
                        config=config_dict,
                        project="CellSeg3D",
                        mode=self.wandb_config.mode,
                    )
                except AttributeError:
                    logger.warning(
                        "Could not initialize wandb."
                        "This might be due to running napari in a folder where there is a directory named 'wandb'."
                        "Aborting, please run napari in a different folder or install wandb. Sorry for the inconvenience."
                    )

            if deterministic_config.enabled:
                set_determinism(
                    seed=deterministic_config.seed
                )  # use_deterministic_algorithms = True causes cuda error

            sys = platform.system()
            logger.debug(sys)
            if sys == "Darwin":  # required for macOS ?
                torch.set_num_threads(1)
                self.log("Number of threads has been set to 1 for macOS")

            self.log(f"config model : {self.config.model_info.name}")
            model_name = model_config.name
            model_class = model_config.get_model()

            ######## Check that labels are semantic, not instance
            check_labels = LoadImaged(keys=["label"])(
                self.config.train_data_dict[0]
            )
            if check_labels["label"].max() > 1:
                self.warn(
                    "Labels are not semantic, but instance. Converting to semantic, this might cause errors."
                )
                self.labels_not_semantic = True
            ########

            if not self.config.sampling:
                data_check = LoadImaged(keys=["image"])(
                    self.config.train_data_dict[0]
                )
                check = data_check["image"].shape
            do_sampling = self.config.sampling
            size = self.config.sample_size if do_sampling else check
            PADDING = utils.get_padding_dim(size)

            model = (
                model_class(input_img_size=PADDING, use_checkpoint=True)
                if provided_model is None
                else provided_model
            )

            device = torch.device(self.config.device)
            model = model.to(device)

            if WANDB_INSTALLED:
                wandb.watch(model, log_freq=100)

            epoch_loss_values = []
            val_metric_values = []

            if len(self.config.train_data_dict) > 1:
                self.train_files, self.val_files = (
                    self.config.train_data_dict[
                        0 : int(
                            len(self.config.train_data_dict)
                            * self.config.training_percent
                        )
                    ],
                    self.config.train_data_dict[
                        int(
                            len(self.config.train_data_dict)
                            * self.config.training_percent
                        ) :
                    ],
                )
            else:
                self.train_files = self.val_files = self.config.train_data_dict
                msg = f"Only one image file was provided : {self.config.train_data_dict[0]['image']}.\n"

                logger.debug(f"SAMPLING is {self.config.sampling}")
                if not self.config.sampling:
                    msg += "Sampling is not in use, the only image provided will be used as the validation file."
                    self.warn(msg)
                else:
                    msg += "Samples for validation will be cropped for the same only volume that is being used for training"

                logger.warning(msg)

            logger.debug(
                f"Data dict from config is {self.config.train_data_dict}"
            )
            logger.debug(f"Train files : {self.train_files}")
            logger.debug(f"Val. files : {self.val_files}")

            if len(self.train_files) == 0:
                raise ValueError("Training dataset is empty")
            if len(self.val_files) == 0:
                raise ValueError("Validation dataset is empty")

            if self.config.do_augmentation:
                train_transforms = (
                    Compose(  # TODO : figure out which ones and values ?
                        [
                            RandShiftIntensityd(keys=["image"], offsets=0.7),
                            Rand3DElasticd(
                                keys=["image", "label"],
                                sigma_range=(0.3, 0.7),
                                magnitude_range=(0.3, 0.7),
                            ),
                            RandFlipd(keys=["image", "label"]),
                            RandRotate90d(keys=["image", "label"]),
                            RandAffined(
                                keys=["image"],
                            ),
                            EnsureTyped(keys=["image"]),
                        ]
                    )
                )
            else:
                train_transforms = EnsureTyped(keys=["image", "label"])

            val_transforms = Compose(
                [
                    EnsureTyped(keys=["image", "label"]),
                ]
            )

            def get_patch_loader_func(num_samples):
                """Returns a function that will be used to extract patches from the images."""
                return Compose(
                    [
                        LoadImaged(keys=["image", "label"]),
                        EnsureChannelFirstd(keys=["image", "label"]),
                        RandSpatialCropSamplesd(
                            keys=["image", "label"],
                            roi_size=(
                                self.config.sample_size
                            ),  # multiply by axis_stretch_factor if anisotropy
                            # max_roi_size=(120, 120, 120),
                            random_size=False,
                            num_samples=num_samples,
                        ),
                        Orientationd(keys=["image", "label"], axcodes="PLI"),
                        SpatialPadd(
                            keys=["image", "label"],
                            spatial_size=(
                                utils.get_padding_dim(self.config.sample_size)
                            ),
                        ),
                        QuantileNormalizationd(keys=["image"]),
                        EnsureTyped(keys=["image"]),
                    ]
                )

            if do_sampling:
                # if there is only one volume, split samples
                # TODO(cyril) : maybe implement something in user config to toggle this behavior
                if len(self.config.train_data_dict) < 2:
                    num_train_samples = ceil(
                        self.config.num_samples * self.config.training_percent
                    )
                    num_val_samples = ceil(
                        self.config.num_samples
                        * (1 - self.config.training_percent)
                    )
                    if num_train_samples < 2:
                        self.log(
                            "WARNING : not enough samples for training. Raising to 2"
                        )
                        num_train_samples = 2
                    if num_val_samples < 2:
                        self.log(
                            "WARNING : not enough samples for validation. Raising to 2"
                        )
                        num_val_samples = 2

                    sample_loader_train = get_patch_loader_func(
                        num_train_samples
                    )
                    sample_loader_eval = get_patch_loader_func(num_val_samples)
                else:
                    num_train_samples = (
                        num_val_samples
                    ) = self.config.num_samples

                    sample_loader_train = get_patch_loader_func(
                        num_train_samples
                    )
                    sample_loader_eval = get_patch_loader_func(num_val_samples)

                logger.debug(f"AMOUNT of train samples : {num_train_samples}")
                logger.debug(
                    f"AMOUNT of validation samples : {num_val_samples}"
                )

                logger.debug("train_ds")
                train_dataset = PatchDataset(
                    data=self.train_files,
                    transform=train_transforms,
                    patch_func=sample_loader_train,
                    samples_per_image=num_train_samples,
                )
                logger.debug("val_ds")
                validation_dataset = PatchDataset(
                    data=self.val_files,
                    transform=val_transforms,
                    patch_func=sample_loader_eval,
                    samples_per_image=num_val_samples,
                )
            else:
                load_whole_images = Compose(
                    [
                        LoadImaged(
                            keys=["image", "label"],
                            # image_only=True,
                            # reader=WSIReader(backend="tifffile")
                        ),
                        EnsureChannelFirstd(keys=["image", "label"]),
                        Orientationd(keys=["image", "label"], axcodes="PLI"),
                        QuantileNormalizationd(keys=["image"]),
                        SpatialPadd(
                            keys=["image", "label"],
                            spatial_size=PADDING,
                        ),
                        EnsureTyped(keys=["image", "label"]),
                    ]
                )
                logger.debug("Cache dataset : train")
                train_dataset = CacheDataset(
                    data=self.train_files,
                    transform=Compose([load_whole_images, train_transforms]),
                )
                logger.debug("Cache dataset : val")
                validation_dataset = CacheDataset(
                    data=self.val_files, transform=load_whole_images
                )
            logger.debug("Dataloader")
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
                collate_fn=pad_list_data_collate,
            )

            validation_loader = DataLoader(
                validation_dataset,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
            )
            logger.debug("\nDone")

            logger.debug("Optimizer")
            optimizer = (
                torch.optim.Adam(model.parameters(), self.config.learning_rate)
                if provided_optimizer is None
                else provided_optimizer
            )

            factor = self.config.scheduler_factor
            if factor >= 1.0:
                self.log(f"Warning : scheduler factor is {factor} >= 1.0")
                self.log("Setting it to 0.5")
                factor = 0.5

            scheduler = (
                torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer=optimizer,
                    mode="min",
                    factor=factor,
                    patience=self.config.scheduler_patience,
                    verbose=VERBOSE_SCHEDULER,
                )
                if provided_scheduler is None
                else provided_scheduler
            )
            dice_metric = DiceMetric(
                include_background=False, reduction="mean", ignore_empty=False
            )

            best_metric = -1
            best_metric_epoch = -1

            # time = utils.get_date_time()
            logger.debug("Weights")

            if weights_config.use_custom:
                if weights_config.use_pretrained:
                    weights_file = model_class.weights_file
                    self.downloader.download_weights(model_name, weights_file)
                    weights = PRETRAINED_WEIGHTS_DIR / Path(weights_file)
                    weights_config.path = weights
                else:
                    weights = str(Path(weights_config.path))

                try:
                    model.load_state_dict(
                        torch.load(
                            weights,
                            map_location=device,
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

            if "cuda" in self.config.device:
                device_id = self.config.device.split(":")[-1]
                self.log("\nUsing GPU :")
                self.log(torch.cuda.get_device_name(int(device_id)))
            else:
                self.log("Using CPU")

            self.log_parameters()

            # device = torch.device(self.config.device)
            self._set_loss_from_config()
            if provided_loss is not None:
                self.loss_function = provided_loss

            # if model_name == "test":
            #     self.quit()
            #     yield TrainingReport(False)

            for epoch in range(self.config.max_epochs):
                # self.log("\n")
                self.log("-" * 10)
                self.log(f"Epoch {epoch + 1}/{self.config.max_epochs}")
                if device.type == "cuda":
                    self.log("Memory Usage:")
                    alloc_mem = round(
                        torch.cuda.memory_allocated(device) / 1024**3, 1
                    )
                    reserved_mem = round(
                        torch.cuda.memory_reserved(device) / 1024**3, 1
                    )
                    self.log(f"Allocated: {alloc_mem}GB")
                    self.log(f"Cached: {reserved_mem}GB")

                model.train()
                epoch_loss = 0
                step = 0
                for batch_data in train_loader:
                    step += 1
                    inputs, labels = (
                        batch_data["image"].to(device),
                        batch_data["label"].to(device),
                    )
                    # logger.debug(f"Inputs shape : {inputs.shape}")
                    # logger.debug(f"Labels shape : {labels.shape}")
                    if self.labels_not_semantic:
                        labels = labels.clamp(0, 1)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    # logger.debug(f"Output dimensions : {outputs.shape}")
                    if outputs.shape[1] > 1:
                        outputs = outputs[
                            :, 1:, :, :
                        ]  # TODO(cyril): adapt if additional channels
                        if len(outputs.shape) < 4:
                            outputs = outputs.unsqueeze(0)
                    # logger.debug(f"Outputs shape : {outputs.shape}")
                    loss = self.loss_function(outputs, labels)

                    if WANDB_INSTALLED:
                        wandb.log({"Training/Loss": loss.item()})

                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.detach().item()
                    self.log(
                        f"* {step}/{len(train_dataset) // train_loader.batch_size}, "
                        f"Train loss: {loss.detach().item():.4f}"
                    )

                    if self._abort_requested:
                        self.log("Aborting training...")
                        model = None
                        del model
                        train_loader = None
                        del train_loader
                        validation_loader = None
                        del validation_loader
                        optimizer = None
                        del optimizer
                        scheduler = None
                        del scheduler
                        if device.type == "cuda":
                            torch.cuda.empty_cache()

                    yield TrainingReport(
                        show_plot=False,
                        weights=model.state_dict(),
                        supervised=True,
                    )

                if WANDB_INSTALLED:
                    wandb.log({"Training/Epoch loss": epoch_loss / step})
                    wandb.log(
                        {
                            "LR/Model learning rate": optimizer.param_groups[
                                0
                            ]["lr"]
                        }
                    )

                epoch_loss /= step
                epoch_loss_values.append(epoch_loss)
                self.log(f"Epoch: {epoch + 1}, Average loss: {epoch_loss:.4f}")

                self.log("Updating scheduler...")
                scheduler.step(epoch_loss)
                self.log(
                    f"Current learning rate: {optimizer.param_groups[0]['lr']}"
                )

                checkpoint_output = []
                eta = (
                    (time.time() - start_time)
                    * (self.config.max_epochs / (epoch + 1) - 1)
                    / 60
                )
                self.log("ETA: " + f"{eta:.2f}" + " minutes")

                if (
                    (epoch + 1) % self.config.validation_interval == 0
                    or epoch + 1 == self.config.max_epochs
                ):
                    model.eval()
                    self.log("Performing validation...")
                    with torch.no_grad():
                        for val_data in validation_loader:
                            val_inputs, val_labels = (
                                val_data["image"].to(device),
                                val_data["label"].to(device),
                            )

                            try:
                                with torch.no_grad():
                                    val_outputs = sliding_window_inference(
                                        val_inputs,
                                        roi_size=size,
                                        sw_batch_size=self.config.batch_size,
                                        predictor=model,
                                        overlap=0.25,
                                        mode="gaussian",
                                        sigma_scale=0.01,
                                        sw_device=self.config.device,
                                        device=self.config.device,
                                        progress=False,
                                    )
                            except Exception as e:
                                self.raise_error(e, "Error during validation")

                            logger.debug(
                                f"val_outputs shape : {val_outputs.shape}"
                            )
                            # val_outputs = model(val_inputs)

                            pred = decollate_batch(val_outputs)
                            labs = decollate_batch(val_labels)
                            # TODO : more parameters/flexibility
                            post_pred = Compose(
                                [
                                    RemapTensor(new_max=1, new_min=0),
                                    Threshold(threshold=0.5),
                                    EnsureType(),
                                ]
                            )  #
                            post_label = EnsureType()

                            output_raw = [
                                RemapTensor(new_max=1, new_min=0)(t)
                                for t in pred
                            ]
                            # output_raw = pred

                            val_outputs = [
                                post_pred(res_tensor) for res_tensor in pred
                            ]

                            val_labels = [
                                post_label(res_tensor) for res_tensor in labs
                            ]

                            # logger.debug(len(val_outputs))
                            # logger.debug(len(val_labels))
                            # dice_test = np.array(
                            #     [
                            #         utils.dice_coeff(i, j)
                            #         for i, j in zip(val_outputs, val_labels)
                            #     ]
                            # )
                            # logger.debug(
                            #     f"TEST VALIDATION Dice score : {dice_test.mean()}"
                            # )

                            dice_metric(y_pred=val_outputs, y=val_labels)

                        checkpoint_output.append(
                            [
                                output_raw[0].detach().cpu(),
                                val_outputs[0].detach().cpu(),
                                val_inputs[0].detach().cpu(),
                                val_labels[0].detach().cpu(),
                            ]
                        )
                        checkpoint_output = [
                            item.numpy()
                            for channel in checkpoint_output
                            for item in channel
                        ]
                        checkpoint_output[3] = checkpoint_output[3].astype(
                            np.uint16
                        )

                        metric = dice_metric.aggregate().detach().item()

                        if WANDB_INSTALLED:
                            wandb.log({"Validation/Dice metric": metric})

                        dice_metric.reset()
                        val_metric_values.append(metric)

                        images_dict = {
                            "Validation output": {
                                "data": checkpoint_output[0],
                                "cmap": "turbo",
                            },
                            "Validation output (discrete)": {
                                "data": checkpoint_output[1],
                                "cmap": "bop blue",
                            },
                            "Validation image": {
                                "data": checkpoint_output[2],
                                "cmap": "inferno",
                            },
                            "Validation labels": {
                                "data": checkpoint_output[3],
                                "cmap": "green",
                            },
                        }

                        train_report = TrainingReport(
                            show_plot=True,
                            epoch=epoch,
                            loss_1_values={"Loss": epoch_loss_values},
                            loss_2_values=val_metric_values,
                            weights=model.state_dict(),
                            images_dict=images_dict,
                            supervised=True,
                        )
                        self.log("Validation completed")
                        yield train_report

                        weights_filename = (
                            f"{model_name}_best_metric"
                            # + f"_epoch_{epoch + 1}" # avoid saving per epoch
                            + ".pth"
                        )

                        if metric > best_metric:
                            best_metric = metric
                            best_metric_epoch = epoch + 1
                            self.log("Saving best metric model")
                            torch.save(
                                model.state_dict(),
                                Path(self.config.results_path_folder)
                                / Path(
                                    weights_filename,
                                ),
                            )
                            self.log("Saving complete")
                        self.log(
                            f"Current epoch: {epoch + 1}, Current mean dice: {metric:.4f}"
                            f"\nBest mean dice: {best_metric:.4f} "
                            f"at epoch: {best_metric_epoch}"
                        )
            self.log("=" * 10)
            self.log(
                f"Train completed, best_metric: {best_metric:.4f} "
                f"at epoch: {best_metric_epoch}"
            )

            if WANDB_INSTALLED:
                wandb.log({"Validation/Best metric": best_metric})
                wandb.log({"Validation/Best metric epoch": best_metric_epoch})

            # Save last checkpoint
            weights_filename = f"{model_name}_latest.pth"
            self.log("Saving last model")
            torch.save(
                model.state_dict(),
                Path(self.config.results_path_folder) / Path(weights_filename),
            )
            self.log("Saving complete, exiting")
            model.to("cpu")
            # clear (V)RAM
            model = None
            del model
            train_loader = None
            del train_loader
            validation_loader = None
            del validation_loader
            optimizer = None
            del optimizer
            scheduler = None
            del scheduler
            if device.type == "cuda":
                torch.cuda.empty_cache()

        except Exception as e:
            self.raise_error(e, "Error in training")
            self.quit()
        finally:
            self.quit()
