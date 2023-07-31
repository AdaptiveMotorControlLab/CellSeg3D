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
    # NormalizeIntensityd,
    Orientationd,
    Rand3DElasticd,
    RandAffined,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    RandSpatialCropSamplesd,
    ScaleIntensityRanged,
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
    # RemapTensord,
    Threshold,
    TrainingReport,
    WeightsDownloader,
)

logger = utils.LOGGER
VERBOSE_SCHEDULER = True
logger.debug(f"PRETRAINED WEIGHT DIR LOCATION : {PRETRAINED_WEIGHTS_DIR}")

try:
    WANDB_INSTALLED = True
except ImportError:
    logger.warning(
        "wandb not installed, wandb config will not be taken into account",
        stacklevel=1,
    )
    WANDB_INSTALLED = False

"""
Writing something to log messages from outside the main thread needs specific care,
Following the instructions in the guides below to have a worker with custom signals,
a custom worker function was implemented.
"""

# https://python-forum.io/thread-31349.html
# https://www.pythoncentral.io/pysidepyqt-tutorial-creating-your-own-signals-and-slots/
# https://napari-staging-site.github.io/guides/stable/threading.html

# TODO list for WNet training :
# 1. Create a custom base worker for training to avoid code duplication
# 2. Create a custom worker for WNet training
# 3. Adapt UI for WNet training (Advanced tab + model choice on first tab)
# 4. Adapt plots and TrainingReport for WNet training
# 5. log_parameters function


class TrainingWorkerBase(GeneratorWorker):
    """A basic worker abstract class, to run training jobs in.
    Contains the minimal common elements required for training models."""

    def __init__(self):
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
        """Sets the log widget for the downloader to output to"""
        self.downloader.log_widget = widget

    def log(self, text):
        """Sends a signal that ``text`` should be logged
        Goes in a Log object, defined in :py:mod:`napari_cellseg3d.interface
        Sends a signal to the main thread to log the text.
        Signal is defined in napari_cellseg3d.workers_utils.LogSignal

        Args:
            text (str): text to logged
        """
        self.log_signal.emit(text)

    def warn(self, warning):
        """Sends a warning to main thread"""
        self.warn_signal.emit(warning)

    def raise_error(self, exception, msg):
        """Sends an error to main thread"""
        logger.error(msg, exc_info=True)
        logger.error(exception, exc_info=True)
        self.error_signal.emit(exception, msg)
        self.errored.emit(exception)
        self.quit()

    @abstractmethod
    def log_parameters(self):
        """Logs the parameters of the training"""
        raise NotImplementedError

    @abstractmethod
    def train(self):
        """Starts a training job"""
        raise NotImplementedError


class WNetTrainingWorker(TrainingWorkerBase):
    """A custom worker to run WNet (unsupervised) training jobs in.
    Inherits from :py:class:`napari.qt.threading.GeneratorWorker` via :py:class:`TrainingWorkerBase`
    """

    def __init__(
        self,
        worker_config: config.WNetTrainingWorkerConfig,
    ):
        super().__init__()
        self.config = worker_config

    def get_patch_dataset(self, train_transforms):
        """Creates a Dataset from the original data using the tifffile library

        Args:
            train_data_dict (dict): dict with the Paths to the directory containing the data

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
        """Creates a Dataset applying some transforms/augmentation on the data using the MONAI library

        Args:
            config (WNetTrainingWorkerConfig): The configuration object

        Returns:
            (tuple): A tuple containing the shape of the data and the dataset
        """
        # train_files = self.create_dataset_dict_no_labs(
        #     volume_directory=self.config.train_volume_directory
        # )
        # self.log(train_files)
        # self.log(len(train_files))
        # self.log(train_files[0])
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

    # def get_scheduler(self, optimizer, verbose=False):
    #     scheduler_name = self.config.scheduler
    #     if scheduler_name == "None":
    #         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #             optimizer,
    #             T_max=100,
    #             eta_min=config.lr - 1e-6,
    #             verbose=verbose,
    #         )
    #
    #     elif scheduler_name == "ReduceLROnPlateau":
    #         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #             optimizer,
    #             mode="min",
    #             factor=schedulers["ReduceLROnPlateau"]["factor"],
    #             patience=schedulers["ReduceLROnPlateau"]["patience"],
    #             verbose=verbose,
    #         )
    #     elif scheduler_name == "CosineAnnealingLR":
    #         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #             optimizer,
    #             T_max=schedulers["CosineAnnealingLR"]["T_max"],
    #             eta_min=schedulers["CosineAnnealingLR"]["eta_min"],
    #             verbose=verbose,
    #         )
    #     elif scheduler_name == "CosineAnnealingWarmRestarts":
    #         scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #             optimizer,
    #             T_0=schedulers["CosineAnnealingWarmRestarts"]["T_0"],
    #             eta_min=schedulers["CosineAnnealingWarmRestarts"]["eta_min"],
    #             T_mult=schedulers["CosineAnnealingWarmRestarts"]["T_mult"],
    #             verbose=verbose,
    #         )
    #     elif scheduler_name == "CyclicLR":
    #         scheduler = torch.optim.lr_scheduler.CyclicLR(
    #             optimizer,
    #             base_lr=schedulers["CyclicLR"]["base_lr"],
    #             max_lr=schedulers["CyclicLR"]["max_lr"],
    #             step_size_up=schedulers["CyclicLR"]["step_size_up"],
    #             mode=schedulers["CyclicLR"]["mode"],
    #             cycle_momentum=False,
    #         )
    #     else:
    #         raise ValueError(f"Scheduler {scheduler_name} not provided")
    #     return scheduler

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
            self.log("Loading patch dataset")
            (data_shape, dataset) = self.get_patch_dataset(train_transforms)
        else:
            self.log("Loading volume dataset")
            (data_shape, dataset) = self.get_dataset(train_transforms)
            # transform = Compose(
            #     [
            #         ToTensor(),
            #         EnsureChannelFirst(channel_dim=0),
            #     ]
            # )
            # dataset = [transform(im) for im in dataset]
            # for data in dataset:
            #     self.log(f"Data shape: {data.shape}")
            #     break
        logger.debug(f"Data shape : {data_shape}")
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=pad_list_data_collate,
        )

        if self.config.eval_volume_dict is not None:
            eval_dataset = self.get_dataset_eval(self.config.eval_volume_dict)

            eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                collate_fn=pad_list_data_collate,
            )
        else:
            eval_dataloader = None
        return dataloader, eval_dataloader, data_shape

    def log_parameters(self):
        self.log("*" * 20)
        self.log("-- Parameters --")
        self.log(f"Device: {self.config.device}")
        self.log(f"Batch size: {self.config.batch_size}")
        self.log(f"Epochs: {self.config.max_epochs}")
        self.log(f"Learning rate: {self.config.learning_rate}")
        self.log(f"Validation interval: {self.config.validation_interval}")
        if self.config.weights_info.custom:
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
        self.log("Training data :")
        [
            self.log(f"{v}")
            for d in self.config.train_data_dict
            for k, v in d.items()
        ]
        if self.config.eval_volume_dict is not None:
            self.log("Validation data :")
            [
                self.log(f"{k}: {v}")
                for d in self.config.eval_volume_dict
                for k, v in d.items()
            ]

    def train(self):
        try:
            if self.config is None:
                self.config = config.WNetTrainingWorkerConfig()
            ##############
            # disable metadata tracking in MONAI
            set_track_meta(False)
            ##############
            # if WANDB_INSTALLED:
            # wandb.init(
            #     config=WANDB_CONFIG, project="WNet-benchmark", mode=WANDB_MODE
            # )

            set_determinism(
                seed=self.config.deterministic_config.seed
            )  # use default seed from NP_MAX
            torch.use_deterministic_algorithms(True, warn_only=True)

            normalize_function = utils.remap_image
            device = self.config.device

            # self.log(f"Using device: {device}")
            self.log_parameters()
            self.log("Initializing training...")
            self.log("Getting the data")

            dataloader, eval_dataloader, data_shape = self._get_data()

            dice_metric = DiceMetric(
                include_background=False, reduction="mean", get_not_nans=False
            )
            ###################################################
            #               Training the model                #
            ###################################################
            self.log("Initializing the model:")
            self.log("- Getting the model")
            # Initialize the model
            model = WNet(
                in_channels=self.config.in_channels,
                out_channels=self.config.out_channels,
                num_classes=self.config.num_classes,
                dropout=self.config.dropout,
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

            # if WANDB_INSTALLED:
            #     wandb.watch(model, log_freq=100)

            if self.config.weights_info.custom:
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

            self.log("- Getting the loss functions")
            # Initialize the Ncuts loss function
            criterionE = SoftNCutsLoss(
                data_shape=data_shape,
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

            # self.log("- getting the learning rate schedulers")
            # Initialize the learning rate schedulers
            # scheduler = get_scheduler(self.config, optimizer)
            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            #     optimizer, mode="min", factor=0.5, patience=10, verbose=True
            # )
            model.train()

            self.log("Ready")
            self.log("Training the model")
            self.log("*" * 20)

            startTime = time.time()
            ncuts_losses = []
            rec_losses = []
            total_losses = []
            best_dice = -1
            dice_values = []

            # Train the model
            for epoch in range(self.config.max_epochs):
                self.log(f"Epoch {epoch + 1} of {self.config.max_epochs}")

                epoch_ncuts_loss = 0
                epoch_rec_loss = 0
                epoch_loss = 0

                for _i, batch in enumerate(dataloader):
                    # raise NotImplementedError("testing")
                    image = batch["image"].to(device)
                    for i in range(image.shape[0]):
                        for j in range(image.shape[1]):
                            image[i, j] = normalize_function(image[i, j])
                    # if self.config.batch_size == 1:
                    #     image = image.unsqueeze(0)
                    # else:
                    #     image = image.unsqueeze(0)
                    #     image = torch.swapaxes(image, 0, 1)

                    # Forward pass
                    enc = model.forward_encoder(image)
                    # Compute the Ncuts loss
                    Ncuts = criterionE(enc, image)
                    epoch_ncuts_loss += Ncuts.item()
                    # if WANDB_INSTALLED:
                    #     wandb.log({"Ncuts loss": Ncuts.item()})

                    # Forward pass
                    enc, dec = model(image)

                    # Compute the reconstruction loss
                    if isinstance(criterionW, nn.MSELoss):
                        reconstruction_loss = criterionW(dec, image)
                    elif isinstance(criterionW, nn.BCELoss):
                        reconstruction_loss = criterionW(
                            torch.sigmoid(dec),
                            utils.remap_image(image, new_max=1),
                        )

                    epoch_rec_loss += reconstruction_loss.item()
                    # if WANDB_INSTALLED:
                    #     wandb.log(
                    #         {"Reconstruction loss": reconstruction_loss.item()}
                    #     )

                    # Backward pass for the reconstruction loss
                    optimizer.zero_grad()
                    alpha = self.config.n_cuts_weight
                    beta = self.config.rec_loss_weight

                    loss = alpha * Ncuts + beta * reconstruction_loss
                    epoch_loss += loss.item()
                    # if WANDB_INSTALLED:
                    #     wandb.log({"Weighted sum of losses": loss.item()})
                    loss.backward(loss)
                    optimizer.step()

                    # if self.config.scheduler == "CosineAnnealingWarmRestarts":
                    #     scheduler.step(epoch + _i / len(dataloader))
                    # if (
                    #         self.config.scheduler == "CosineAnnealingLR"
                    #         or self.config.scheduler == "CyclicLR"
                    # ):
                    #     scheduler.step()

                    yield TrainingReport(
                        show_plot=False, weights=model.state_dict()
                    )

                ncuts_losses.append(epoch_ncuts_loss / len(dataloader))
                rec_losses.append(epoch_rec_loss / len(dataloader))
                total_losses.append(epoch_loss / len(dataloader))

                if eval_dataloader is None:
                    try:
                        enc_out = enc[0].detach().cpu().numpy()
                        dec_out = dec[0].detach().cpu().numpy()
                        image = image[0].detach().cpu().numpy()

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
                                "data": np.squeeze(image),
                                "cmap": "inferno",
                            },
                        }

                        yield TrainingReport(
                            show_plot=True,
                            epoch=epoch,
                            loss_1_values={"SoftNCuts": ncuts_losses},
                            loss_2_values=rec_losses,
                            weights=model.state_dict(),
                            images_dict=images_dict,
                        )
                    except TypeError:
                        pass

                # if WANDB_INSTALLED:
                #     wandb.log({"Ncuts loss_epoch": ncuts_losses[-1]})
                #     wandb.log({"Reconstruction loss_epoch": rec_losses[-1]})
                #     wandb.log({"Sum of losses_epoch": total_losses[-1]})
                # wandb.log({"epoch": epoch})
                # wandb.log({"learning_rate model": optimizerW.param_groups[0]["lr"]})
                # wandb.log({"learning_rate encoder": optimizerE.param_groups[0]["lr"]})
                # wandb.log({"learning_rate model": optimizer.param_groups[0]["lr"]})

                self.log(f"Ncuts loss: {ncuts_losses[-1]:.5f}")
                self.log(f"Reconstruction loss: {rec_losses[-1]:.5f}")
                self.log(f"Weighted sum of losses: {total_losses[-1]:.5f}")
                if epoch > 0:
                    self.log(
                        f"Ncuts loss difference: {ncuts_losses[-1] - ncuts_losses[-2]:.5f}"
                    )
                    self.log(
                        f"Reconstruction loss difference: {rec_losses[-1] - rec_losses[-2]:.5f}"
                    )
                    self.log(
                        f"Weighted sum of losses difference: {total_losses[-1] - total_losses[-2]:.5f}"
                    )

                # Update the learning rate
                # if self.config.scheduler == "ReduceLROnPlateau":
                #     # schedulerE.step(epoch_ncuts_loss)
                #     # schedulerW.step(epoch_rec_loss)
                #     scheduler.step(epoch_rec_loss)
                if (
                    eval_dataloader is not None
                    and (epoch + 1) % self.config.validation_interval == 0
                ):
                    model.eval()
                    self.log("Validating...")
                    with torch.no_grad():
                        for _k, val_data in enumerate(eval_dataloader):
                            val_inputs, val_labels = (
                                val_data["image"].to(device),
                                val_data["label"].to(device),
                            )

                            # normalize val_inputs across channels
                            for i in range(val_inputs.shape[0]):
                                for j in range(val_inputs.shape[1]):
                                    val_inputs[i][j] = normalize_function(
                                        val_inputs[i][j]
                                    )
                            logger.debug(
                                f"Val inputs shape: {val_inputs.shape}"
                            )
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
                            val_outputs = AsDiscrete(threshold=0.5)(
                                val_outputs
                            )
                            logger.debug(
                                f"Val outputs shape: {val_outputs.shape}"
                            )
                            logger.debug(
                                f"Val labels shape: {val_labels.shape}"
                            )
                            logger.debug(
                                f"Val decoder outputs shape: {val_decoder_outputs.shape}"
                            )

                            dices = []
                            for channel in range(val_outputs.shape[1]):
                                dices.append(
                                    utils.dice_coeff(
                                        y_pred=val_outputs[
                                            0, channel : (channel + 1), :, :, :
                                        ],
                                        y_true=val_labels[0],
                                    )
                                )
                            logger.debug(f"DICE COEFF: {dices}")
                            max_dice_channel = torch.argmax(
                                torch.Tensor(dices)
                            )
                            logger.debug(
                                f"MAX DICE CHANNEL: {max_dice_channel}"
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

                        # aggregate the final mean dice result
                        metric = dice_metric.aggregate().item()
                        dice_values.append(metric)
                        self.log(f"Validation Dice score: {metric:.3f}")
                        if best_dice < metric <= 1:
                            best_dice = metric
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

                        # if WANDB_INSTALLED:
                        # log validation dice score for each validation round
                        # wandb.log({"val/dice_metric": metric})

                        dec_out_val = (
                            val_decoder_outputs[0].detach().cpu().numpy()
                        )
                        enc_out_val = val_outputs[0].detach().cpu().numpy()
                        lab_out_val = val_labels[0].detach().cpu().numpy()
                        val_in = val_inputs[0].detach().cpu().numpy()

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

                        yield TrainingReport(
                            epoch=epoch,
                            loss_1_values={
                                "SoftNCuts": ncuts_losses,
                                "Dice metric": dice_values,
                            },
                            loss_2_values=rec_losses,
                            weights=model.state_dict(),
                            images_dict=display_dict,
                        )

                        # reset the status for next validation round
                        dice_metric.reset()

                eta = (
                    (time.time() - startTime)
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
            if best_dice > -1:
                self.log(f"Best dice metric : {best_dice}")
            # if WANDB_INSTALLED and self.config.eval_volume_directory is not None:
            #     wandb.log(
            #         {
            #             "best_dice_metric": best_dice,
            #             "best_metric_epoch": best_dice_epoch,
            #         }
            #     )
            self.log("*" * 50)

            # Save the model

            print(
                "Saving the model to: ",
                self.config.results_path_folder + "/wnet.pth",
            )
            torch.save(
                model.state_dict(),
                self.config.results_path_folder + "/wnet.pth",
            )

            # if WANDB_INSTALLED:
            #     model_artifact = wandb.Artifact(
            #         "WNet",
            #         type="model",
            #         description="WNet benchmark",
            #         metadata=dict(WANDB_CONFIG),
            #     )
            #     model_artifact.add_file(self.config.save_model_path)
            #     wandb.log_artifact(model_artifact)

            return ncuts_losses, rec_losses, model
        except Exception as e:
            msg = f"Training failed with exception: {e}"
            self.log(msg)
            self.raise_error(e, msg)
            self.quit()
            raise e


class SupervisedTrainingWorker(TrainingWorkerBase):
    """A custom worker to run supervised training jobs in.
    Inherits from :py:class:`napari.qt.threading.GeneratorWorker` via :py:class:`TrainingWorkerBase`
    """

    def __init__(
        self,
        worker_config: config.SupervisedTrainingWorkerConfig,
    ):
        """Initializes a worker for inference with the arguments needed by the :py:func:`~train` function. Note: See :py:func:`~train`

        Args:
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

    def set_loss_from_config(self):
        try:
            self.loss_function = self.loss_dict[self.config.loss_function]
        except KeyError as e:
            self.raise_error(e, "Loss function not found, aborting job")
        return self.loss_function

    def log_parameters(self):
        self.log("-" * 20)
        self.log("Parameters summary :\n")

        self.log(
            f"Percentage of dataset used for validation : {self.config.validation_percent * 100}%"
        )

        self.log("-" * 10)
        self.log("Training files :\n")
        [
            self.log(f"{Path(train_file['image']).name}\n")
            for train_file in self.train_files
        ]
        self.log("-" * 10)
        self.log("Validation files :\n")
        [
            self.log(f"{Path(val_file['image']).name}\n")
            for val_file in self.val_files
        ]
        self.log("-" * 10)

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

        if not self.config.weights_info.use_pretrained:
            self.log(f"Using weights from : {self.config.weights_info.path}")
            if self._weight_error:
                self.log(
                    ">>>>>>>>>>>>>>>>>\n"
                    "WARNING:\nChosen weights were incompatible with the model,\n"
                    "the model will be trained from random weights\n"
                    "<<<<<<<<<<<<<<<<<\n"
                )

        # self.log("\n")
        self.log("-" * 20)

    def train(self):
        """Trains the PyTorch model for the given number of epochs, with the selected model and data,
        using the chosen batch size, validation interval, loss function, and number of samples.
        Will perform validation once every :py:obj:`val_interval` and save results if the mean dice is better

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

            if not self.config.sampling:
                data_check = LoadImaged(keys=["image"])(
                    self.config.train_data_dict[0]
                )
                check = data_check["image"].shape

            do_sampling = self.config.sampling

            size = self.config.sample_size if do_sampling else check

            PADDING = utils.get_padding_dim(size)
            model = model_class(input_img_size=PADDING, use_checkpoint=True)
            model = model.to(self.config.device)

            epoch_loss_values = []
            val_metric_values = []

            if len(self.config.train_data_dict) > 1:
                self.train_files, self.val_files = (
                    self.config.train_data_dict[
                        0 : int(
                            len(self.config.train_data_dict)
                            * self.config.validation_percent
                        )
                    ],
                    self.config.train_data_dict[
                        int(
                            len(self.config.train_data_dict)
                            * self.config.validation_percent
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
                                keys=["image", "label"],
                            ),
                            EnsureTyped(keys=["image", "label"]),
                        ]
                    )
                )
            else:
                train_transforms = EnsureTyped(keys=["image", "label"])

            val_transforms = Compose(
                [
                    # LoadImaged(keys=["image", "label"]),
                    # EnsureChannelFirstd(keys=["image", "label"]),
                    EnsureTyped(keys=["image", "label"]),
                ]
            )

            # self.log("Loading dataset...\n")
            def get_loader_func(num_samples):
                return Compose(
                    [
                        LoadImaged(keys=["image", "label"]),
                        EnsureChannelFirstd(keys=["image", "label"]),
                        QuantileNormalizationd(keys=["image"]),
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
                        EnsureTyped(keys=["image", "label"]),
                    ]
                )

            if do_sampling:
                # if there is only one volume, split samples
                # TODO(cyril) : maybe implement something in user config to toggle this behavior
                if len(self.config.train_data_dict) < 2:
                    num_train_samples = ceil(
                        self.config.num_samples
                        * self.config.validation_percent
                    )
                    num_val_samples = ceil(
                        self.config.num_samples
                        * (1 - self.config.validation_percent)
                    )
                    sample_loader_train = get_loader_func(num_train_samples)
                    sample_loader_eval = get_loader_func(num_val_samples)
                else:
                    num_train_samples = (
                        num_val_samples
                    ) = self.config.num_samples

                    sample_loader_train = get_loader_func(num_train_samples)
                    sample_loader_eval = get_loader_func(num_val_samples)

                logger.debug(f"AMOUNT of train samples : {num_train_samples}")
                logger.debug(
                    f"AMOUNT of validation samples : {num_val_samples}"
                )

                logger.debug("train_ds")
                train_ds = PatchDataset(
                    data=self.train_files,
                    transform=train_transforms,
                    patch_func=sample_loader_train,
                    samples_per_image=num_train_samples,
                )
                logger.debug("val_ds")
                val_ds = PatchDataset(
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
                train_ds = CacheDataset(
                    data=self.train_files,
                    transform=Compose(load_whole_images, train_transforms),
                )
                logger.debug("Cache dataset : val")
                val_ds = CacheDataset(
                    data=self.val_files, transform=load_whole_images
                )
            logger.debug("Dataloader")
            train_loader = DataLoader(
                train_ds,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=2,
                collate_fn=pad_list_data_collate,
            )

            val_loader = DataLoader(
                val_ds, batch_size=self.config.batch_size, num_workers=2
            )
            logger.info("\nDone")

            logger.debug("Optimizer")
            optimizer = torch.optim.Adam(
                model.parameters(), self.config.learning_rate
            )

            factor = self.config.scheduler_factor
            if factor >= 1.0:
                self.log(f"Warning : scheduler factor is {factor} >= 1.0")
                self.log("Setting it to 0.5")
                factor = 0.5

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                mode="min",
                factor=factor,
                patience=self.config.scheduler_patience,
                verbose=VERBOSE_SCHEDULER,
            )
            dice_metric = DiceMetric(
                include_background=True, reduction="mean", ignore_empty=False
            )

            best_metric = -1
            best_metric_epoch = -1

            # time = utils.get_date_time()
            logger.debug("Weights")

            if weights_config.custom:
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

            if "cuda" in self.config.device:
                device_id = self.config.device.split(":")[-1]
                self.log("\nUsing GPU :")
                self.log(torch.cuda.get_device_name(int(device_id)))
            else:
                self.log("Using CPU")

            self.log_parameters()

            device = torch.device(self.config.device)
            self.set_loss_from_config()

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

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    # self.log(f"Output dimensions : {outputs.shape}")
                    if outputs.shape[1] > 1:
                        outputs = outputs[
                            :, 1:, :, :
                        ]  # TODO(cyril): adapt if additional channels
                        if len(outputs.shape) < 4:
                            outputs = outputs.unsqueeze(0)
                    loss = self.loss_function(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.detach().item()
                    self.log(
                        f"* {step}/{len(train_ds) // train_loader.batch_size}, "
                        f"Train loss: {loss.detach().item():.4f}"
                    )
                    yield TrainingReport(
                        show_plot=False, weights=model.state_dict()
                    )

                # return

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
                        for val_data in val_loader:
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
            # val_ds = None
            # train_ds = None
            # val_loader = None
            # train_loader = None
            # torch.cuda.empty_cache()

        except Exception as e:
            self.raise_error(e, "Error in training")
            self.quit()
        finally:
            self.quit()
