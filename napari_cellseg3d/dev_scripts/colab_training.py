"""Script to run WNet training in Google Colab."""

import time
from pathlib import Path
from typing import TYPE_CHECKING

from monai.data import CacheDataset

# MONAI
from monai.metrics import DiceMetric
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
)

# local
from napari_cellseg3d import config, utils
from napari_cellseg3d.code_models.worker_training import WNetTrainingWorker
from napari_cellseg3d.code_models.workers_utils import (
    PRETRAINED_WEIGHTS_DIR,
)

if TYPE_CHECKING:
    from monai.data import DataLoader

logger = utils.LOGGER
VERBOSE_SCHEDULER = True
logger.debug(f"PRETRAINED WEIGHT DIR LOCATION : {PRETRAINED_WEIGHTS_DIR}")


class LogFixture:
    """Fixture for napari-less logging, replaces napari_cellseg3d.interface.Log in model_workers.

    This allows to redirect the output of the workers to stdout instead of a specialized widget.
    """

    def __init__(self):
        """Creates a LogFixture object."""
        super(LogFixture, self).__init__()

    def print_and_log(self, text, printing=None):
        """Prints and logs text."""
        print(text)

    def warn(self, warning):
        """Logs warning."""
        logger.warning(warning)

    def error(self, e):
        """Logs error."""
        raise (e)


class WNetTrainingWorkerColab(WNetTrainingWorker):
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
        super().__init__(worker_config)
        super().__init__(worker_config)
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

        if len(first_volume_shape) != 3:
            raise ValueError(
                f"Expected 3D volumes, got {len(first_volume_shape)} dimensions"
            )

        # Transforms to be applied to each volume
        load_single_images = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(
                    keys=["image"],
                    channel_dim="no_channel",
                    strict_check=False,
                ),
                Orientationd(keys=["image"], axcodes="PLI"),
                # SpatialPadd(
                #     keys=["image"],
                #     spatial_size=(utils.get_padding_dim(first_volume_shape)),
                # ),
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


def get_colab_worker(
    worker_config: config.WNetTrainingWorkerConfig,
    wandb_config: config.WandBConfig,
):
    """Train a WNet model in Google Colab.

    Args:
        worker_config (config.WNetTrainingWorkerConfig): config for the training worker
        wandb_config (config.WandBConfig): config for wandb
    """
    log = LogFixture()
    worker = WNetTrainingWorkerColab(worker_config, wandb_config)

    worker.log_signal.connect(log.print_and_log)
    worker.warn_signal.connect(log.warn)
    worker.error_signal.connect(log.error)

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
