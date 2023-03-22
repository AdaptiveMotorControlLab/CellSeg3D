import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import napari
import numpy as np

from napari_cellseg3d.code_models.model_instance_seg import ConnectedComponents
from napari_cellseg3d.code_models.model_instance_seg import InstanceMethod
from napari_cellseg3d.code_models.model_instance_seg import VoronoiOtsu
from napari_cellseg3d.code_models.model_instance_seg import Watershed


# from napari_cellseg3d.models import model_TRAILMAP as TRAILMAP
from napari_cellseg3d.code_models.models import model_SegResNet as SegResNet
from napari_cellseg3d.code_models.models import model_SwinUNetR as SwinUNetR
from napari_cellseg3d.code_models.models import (
    model_TRAILMAP_MS as TRAILMAP_MS,
)
from napari_cellseg3d.code_models.models import model_VNet as VNet
from napari_cellseg3d.utils import LOGGER

logger = LOGGER

# TODO(cyril) DOCUMENT !!! and add default values
# TODO(cyril) add JSON load/save

MODEL_LIST = {
    "SegResNet": SegResNet_,
    "VNet": VNet_,
    # "TRAILMAP": TRAILMAP,
    "TRAILMAP_MS": TRAILMAP_MS_,
    "SwinUNetR": SwinUNETR_,
    "WNet": WNet_,
    # "test" : DO NOT USE, reserved for testing
}


WEIGHTS_DIR = str(
    Path(__file__).parent.resolve() / Path("code_models/models/pretrained")
)


################
# Review


@dataclass
class ReviewConfig:
    image: np.array = None
    labels: np.array = None
    csv_path: str = Path.home() / Path("cellseg3d/review")
    model_name: str = ""
    new_csv: bool = True
    filetype: str = ".tif"
    zoom_factor: List[int] = None


@dataclass  # TODO create custom reader for JSON to load project
class ReviewSession:
    project_name: str
    image_path: str
    labels_path: str
    csv_path: str
    aniso_zoom: List[int]
    time_taken: datetime.timedelta


################
# Model & weights


@dataclass
class ModelInfo:
    """Dataclass recording model info
    Args:
        name (str): name of the model
        model_input_size (Optional[List[int]]): input size of the model
        num_classes (int): number of classes for the model
    """

    name: str = next(iter(MODEL_LIST))
    model_input_size: Optional[List[int]] = None
    num_classes: int = 2

    def get_model(self):
        try:
            return MODEL_LIST[self.name]
        except KeyError as e:
            msg = f"Model {self.name} is not defined"
            logger.warning(msg)
            logger.warning(msg)
            raise KeyError from e

    @staticmethod
    def get_model_name_list():
        logger.info("Model list :")
        for model_name in MODEL_LIST:
            logger.info(f" * {model_name}")
        return MODEL_LIST.keys()


@dataclass
class WeightsInfo:
    path: str = PRETRAINED_WEIGHTS_DIR
    custom: bool = False
    use_pretrained: Optional[bool] = False


################
# Post processing & instance segmentation


@dataclass
class Thresholding:
    enabled: bool = True
    threshold_value: float = 0.8


@dataclass
class Zoom:
    enabled: bool = True
    zoom_values: List[float] = None


@dataclass
class InstanceSegConfig:
    enabled: bool = False
    method: InstanceMethod = None


@dataclass
class PostProcessConfig:
    """Class to record params for post processing

    Args:
        zoom (Zoom): zoom config
        thresholding (Thresholding): thresholding config
        instance (InstanceSegConfig): instance segmentation config
    """

    zoom: Zoom = Zoom()
    thresholding: Thresholding = Thresholding()
    instance: InstanceSegConfig = InstanceSegConfig()


@dataclass
class CRFConfig:
    """
    Class to record params for CRF
    """

    sa: float = 10
    sb: float = 5
    sg: float = 1
    w1: float = 10
    w2: float = 5
    n_iters: int = 5


################
# Inference configs


@dataclass
class SlidingWindowConfig:
    window_size: int = None
    window_overlap: float = 0.25

    def is_enabled(self):
        return self.window_size is not None


@dataclass
class InfererConfig:
    """Class to record params for Inferer plugin

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
    """Class to record configuration for Inference job

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
    results_path: str = str(Path.home() / Path("cellseg3d/inference"))
    filetype: str = ".tif"
    keep_on_cpu: bool = False
    compute_stats: bool = False
    post_process_config: PostProcessConfig = PostProcessConfig()
    sliding_window_config: SlidingWindowConfig = SlidingWindowConfig()
    use_crf: bool = False
    crf_config: CRFConfig = CRFConfig()

    images_filepaths: str = None
    layer: napari.layers.Layer = None


################
# Training configs


@dataclass
class DeterministicConfig:
    """Class to record deterministic config"""

    enabled: bool = False
    seed: int = 23498


@dataclass
class TrainerConfig:
    """Class to record trainer plugin config"""

    save_as_zip: bool = False


@dataclass
class TrainingWorkerConfig:
    """Class to record config for Trainer plugin"""

    device: str = "cpu"
    model_info: ModelInfo = None
    weights_info: WeightsInfo = None
    train_data_dict: dict = None
    validation_percent: float = 0.8
    max_epochs: int = 5
    loss_function: callable = None
    learning_rate: np.float64 = 1e-3
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    validation_interval: int = 2
    batch_size: int = 1
    results_path_folder: str = str(Path.home() / Path("cellseg3d/training"))
    sampling: bool = False
    num_samples: int = 2
    sample_size: List[int] = None
    do_augmentation: bool = True
    deterministic_config: DeterministicConfig = DeterministicConfig()


################
# CRF config for WNet
################


@dataclass
class WNetCRFConfig:
    "Class to store parameters of WNet CRF post processing"

    # CRF
    sa = 10  # 50
    sb = 10
    sg = 1
    w1 = 10  # 50
    w2 = 10
    n_iter = 5
