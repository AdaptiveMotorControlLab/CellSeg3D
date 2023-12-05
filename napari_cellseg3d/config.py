"""Module to store configuration parameters for napari_cellseg3d."""
import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import napari
import numpy as np

from napari_cellseg3d.code_models.instance_segmentation import InstanceMethod

# from napari_cellseg3d.models import model_TRAILMAP as TRAILMAP
from napari_cellseg3d.code_models.models.model_SegResNet import SegResNet_
from napari_cellseg3d.code_models.models.model_SwinUNetR import SwinUNETR_
from napari_cellseg3d.code_models.models.model_TRAILMAP_MS import TRAILMAP_MS_
from napari_cellseg3d.code_models.models.model_VNet import VNet_
from napari_cellseg3d.code_models.models.model_WNet import WNet_
from napari_cellseg3d.utils import LOGGER

logger = LOGGER

# TODO(cyril) add JSON load/save

MODEL_LIST = {
    "SegResNet": SegResNet_,
    "VNet": VNet_,
    "TRAILMAP_MS": TRAILMAP_MS_,
    "SwinUNetR": SwinUNETR_,
    "WNet": WNet_,
    # "TRAILMAP": TRAILMAP,
    # "test" : DO NOT USE, reserved for testing
}

PRETRAINED_WEIGHTS_DIR = str(
    Path(__file__).parent.resolve() / Path("code_models/models/pretrained")
)


################
#     Review   #
################


@dataclass
class ReviewConfig:
    """Class to record params for Review plugin.

    Args:
        image (np.array): image to review
        labels (np.array): labels to review
        csv_path (str): path to csv to save results
        model_name (str): name of the model (added to csv name)
        new_csv (bool): whether to create a new csv
        filetype (str): filetype to read & write review images
        zoom_factor (List[int]): zoom factor to apply to image & labels, if selected
    """

    image: np.array = None
    labels: np.array = None
    csv_path: str = Path.home() / "cellseg3d" / "review"
    model_name: str = ""
    new_csv: bool = True
    filetype: str = ".tif"
    zoom_factor: List[int] = None


@dataclass  # TODO create custom reader for JSON to load project
class ReviewSession:
    """Class to record params for Review session.

    Args:
        project_name (str): name of the project
        image_path (str): path to images
        labels_path (str): path to labels
        csv_path (str): path to csv
        aniso_zoom (List[int]): anisotropy zoom
        time_taken (datetime.timedelta): time taken to review
    """

    project_name: str
    image_path: str
    labels_path: str
    csv_path: str
    aniso_zoom: List[int]
    time_taken: datetime.timedelta


###################
# Model & weights #
###################


@dataclass
class ModelInfo:
    """Dataclass recording supervised models info.

    Args:
        name (str): name of the model
        model_input_size (Optional[List[int]]): input size of the model
        num_classes (int): number of classes for the model.
    """

    name: str = next(iter(MODEL_LIST))
    model_input_size: Optional[
        List[int]
    ] = None  # only used by SegResNet and SwinUNETR
    num_classes: int = 2  # only used by WNets

    def get_model(self):
        """Return model from model list."""
        try:
            return MODEL_LIST[self.name]
        except KeyError as e:
            msg = f"Model {self.name} is not defined"
            logger.warning(msg)
            logger.warning(msg)
            raise KeyError from e

    @staticmethod
    def get_model_name_list():
        """Return list of model names."""
        logger.info("Model list :")
        for model_name in MODEL_LIST:
            logger.info(f" * {model_name}")
        return MODEL_LIST.keys()


@dataclass
class WeightsInfo:
    """Class to record params for weights.

    Args:
        path (Optional[str]): path to weights
        use_custom (Optional[bool]): whether to use custom weights
        use_pretrained (Optional[bool]): whether to use pretrained weights
    """

    path: Optional[str] = PRETRAINED_WEIGHTS_DIR
    use_pretrained: Optional[bool] = False
    use_custom: Optional[bool] = False


#############################################
# Post processing & instance segmentation   #
#############################################

# Utils


@dataclass
class Thresholding:
    """Class to record params for thresholding."""

    enabled: bool = False
    threshold_value: float = 0.8


@dataclass
class Zoom:
    """Class to record params for zoom."""

    enabled: bool = False
    zoom_values: List[float] = None


@dataclass
class InstanceSegConfig:
    """Class to record params for instance segmentation."""

    enabled: bool = False
    method: InstanceMethod = None


# Workers
@dataclass
class PostProcessConfig:
    """Class to record params for post processing.

    Args:
        zoom (Zoom): zoom config
        thresholding (Thresholding): thresholding config
        instance (InstanceSegConfig): instance segmentation config
    """

    zoom: Zoom = Zoom()
    thresholding: Thresholding = Thresholding()
    instance: InstanceSegConfig = InstanceSegConfig()
    artifact_removal: bool = False
    artifact_removal_size: int = 500


@dataclass
class CRFConfig:
    """Class to record params for CRF.

    Args:
        sa (float): alpha standard deviation, the scale of the spatial part of the appearance/bilateral kernel.
        sb (float): beta standard deviation, the scale of the color part of the appearance/bilateral kernel.
        sg (float): gamma standard deviation, the scale of the smoothness/gaussian kernel.
        w1 (float): weight of the appearance/bilateral kernel.
        w2 (float): weight of the smoothness/gaussian kernel.
        n_iter (int, optional): Number of iterations for the CRF post-processing step. Defaults to 5.
    """

    sa: float = 10
    sb: float = 5
    sg: float = 1
    w1: float = 10
    w2: float = 5
    n_iters: int = 5


#####################
# Inference configs #
#####################


@dataclass
class SlidingWindowConfig:
    """Class to record params for sliding window inference."""

    window_size: int = None
    window_overlap: float = 0.25

    def is_enabled(self):
        """Return True if sliding window is enabled."""
        return self.window_size is not None


@dataclass
class InfererConfig:
    """Class to record params for Inferer plugin.

    Args:
        model_info (ModelInfo): model info
        show_results (bool): show results in napari
        show_results_count (int): number of results to show
        show_original (bool): show original image in napari
        anisotropy_resolution (List[int]): anisotropy resolution
    """

    model_info: ModelInfo = None
    show_results: bool = False
    show_results_count: int = 5
    show_original: bool = True
    anisotropy_resolution: List[int] = None


@dataclass
class InferenceWorkerConfig:
    """Class to record configuration for Inference job.

    Args:
        device (str): device to use for inference
        model_info (ModelInfo): model info
        weights_config (WeightsInfo): weights info
        results_path (str): path to save results
        filetype (str): filetype to save results
        keep_on_cpu (bool): keep results on cpu
        compute_stats (bool): compute stats
        post_process_config (PostProcessConfig): post processing config
        sliding_window_config (SlidingWindowConfig): sliding window config
        images_filepaths (str): path to images to infer
        layer (napari.layers.Layer): napari layer to infer on
    """

    device: str = "cpu"
    model_info: ModelInfo = ModelInfo()
    weights_config: WeightsInfo = WeightsInfo()
    results_path: str = str(Path.home() / "cellseg3d" / "inference")
    filetype: str = ".tif"
    keep_on_cpu: bool = False
    compute_stats: bool = False
    post_process_config: PostProcessConfig = PostProcessConfig()
    sliding_window_config: SlidingWindowConfig = SlidingWindowConfig()
    use_crf: bool = False
    crf_config: CRFConfig = CRFConfig()

    images_filepaths: List[str] = None
    layer: napari.layers.Layer = None


####################
# Training configs #
####################


@dataclass
class DeterministicConfig:
    """Class to record deterministic config."""

    enabled: bool = True
    seed: int = 34936339  # default seed from NP_MAX


@dataclass
class TrainerConfig:
    """Class to record trainer plugin config."""

    save_as_zip: bool = False


@dataclass
class TrainingWorkerConfig:
    """General class to record config for training.

    Args:
        device (str): device to use for training
        max_epochs (int): max number of epochs
        learning_rate (np.float64): learning rate
        validation_interval (int): validation interval
        batch_size (int): batch size
        deterministic_config (DeterministicConfig): deterministic config
        scheduler_factor (float): scheduler factor
        scheduler_patience (int): scheduler patience
        weights_info (WeightsInfo): weights info
        results_path_folder (str): path to save results
        sampling (bool): whether to sample data into patches
        num_samples (int): number of patches
        sample_size (List[int]): patch size
        do_augmentation (bool): whether to do augmentation
        num_workers (int): number of workers
        train_data_dict (dict): dict of train data as {"image": np.array, "labels": np.array}
    """

    # model params
    device: str = "cpu"
    max_epochs: int = 50
    learning_rate: np.float64 = 1e-3
    validation_interval: int = 2
    batch_size: int = 1
    deterministic_config: DeterministicConfig = DeterministicConfig()
    scheduler_factor: float = 0.5
    scheduler_patience: int = 10
    weights_info: WeightsInfo = WeightsInfo()
    # data params
    results_path_folder: str = str(Path.home() / "cellseg3d" / "training")
    sampling: bool = False
    num_samples: int = 2
    sample_size: List[int] = None
    do_augmentation: bool = True
    num_workers: int = 4
    train_data_dict: dict = None


@dataclass
class SupervisedTrainingWorkerConfig(TrainingWorkerConfig):
    """Class to record config for Trainer plugin.

    Args:
        model_info (ModelInfo): model info
        loss_function (callable): loss function
        validation_percent (float): validation percent
    """

    model_info: ModelInfo = None
    loss_function: callable = None
    training_percent: float = 0.8


@dataclass
class WNetTrainingWorkerConfig(TrainingWorkerConfig):
    """Class to record config for WNet worker.

    Args:
        in_channels (int): encoder input channels
        out_channels (int): decoder (reconstruction) output channels
        num_classes (int): encoder output channels
        dropout (float): dropout
        learning_rate (np.float64): learning rate
        use_clipping (bool): use gradient clipping
        clipping (float): clipping value
        weight_decay (float): weight decay
        intensity_sigma (float): intensity sigma
        spatial_sigma (float): spatial sigma
        radius (int): pixel radius for loss computation; might be overriden depending on data shape
        reconstruction_loss (str): reconstruction loss (MSE or BCE)
        n_cuts_weight (float): weight for NCuts loss
        rec_loss_weight (float): weight for reconstruction loss. Must be adjusted depending on images; compare to NCuts loss value
        train_data_dict (dict): dict of train data as {"image": np.array, "labels": np.array}
        eval_volume_dict (str): dict of eval volume (optional)
        eval_batch_size (int): eval batch size (optional)
    """

    # model params
    in_channels: int = 1
    out_channels: int = 1
    num_classes: int = 2
    dropout: float = 0.65
    learning_rate: np.float64 = 2e-5
    use_clipping: bool = False
    clipping: float = 1.0
    weight_decay: float = 0.01  # 1e-5
    # NCuts loss params
    intensity_sigma: float = 1.0
    spatial_sigma: float = 4.0
    radius: int = 2
    # reconstruction loss params
    reconstruction_loss: str = "MSE"  # or "BCE"
    # summed losses weights
    n_cuts_weight: float = 0.5
    rec_loss_weight: float = 0.5 / 100
    # normalization params
    # normalizing_function: callable = remap_image # FIXME: call directly in worker, not a param
    # data params
    train_data_dict: dict = None
    eval_volume_dict: str = None
    eval_batch_size: int = 1


################
# WandB config #
################
@dataclass
class WandBConfig:
    """Class to record parameters for WandB."""

    mode: str = "online"  # disabled, online, offline
    save_model_artifact: bool = False
