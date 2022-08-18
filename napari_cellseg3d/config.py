from dataclasses import dataclass
from typing import List
from typing import Optional
import warnings

import napari
import numpy as np

from napari_cellseg3d.models import model_SegResNet as SegResNet
from napari_cellseg3d.models import model_SwinUNetR as SwinUNetR

# from napari_cellseg3d.models import model_TRAILMAP as TRAILMAP
from napari_cellseg3d.models import model_VNet as VNet
from napari_cellseg3d.models import model_TRAILMAP_MS as TRAILMAP_MS

from napari_cellseg3d.model_instance_seg import binary_connected
from napari_cellseg3d.model_instance_seg import binary_watershed

# TODO DOCUMENT !!! and add default values

MODEL_LIST = {
    "VNet": VNet,
    "SegResNet": SegResNet,
    # "TRAILMAP": TRAILMAP,
    "TRAILMAP_MS": TRAILMAP_MS,
    "SwinUNetR": SwinUNetR,
}

INSTANCE_SEGMENTATION_METHOD_LIST = {
    "Watershed": binary_watershed,
    "Connected components": binary_connected,
}


@dataclass
class ModelInfo:
    """Dataclass recording model info :
    - name (str): name of the model"""

    name: str
    model_input_size: Optional[List[int]] = None

    def get_model(self):
        try:
            return MODEL_LIST[self.name]
        except KeyError as e:
            warnings.warn(f"Model {self.name} is not defined")
            raise KeyError(e)

    @staticmethod
    def get_model_name_list():
        print(f"Model list :\n" + [f"{name}\n" for name in MODEL_LIST.keys()])
        return MODEL_LIST.keys()


@dataclass
class WeightsInfo:
    path: str = None
    custom: bool = False
    use_pretrained: Optional[bool] = False


################
# Post processing & instance segmentation


@dataclass
class Thresholding:
    enabled: bool = True
    threshold_value: float = 1.0


@dataclass
class Zoom:
    enabled: bool = True
    zoom_values: List[float] = None


@dataclass
class InstanceSegConfig:
    enabled: bool = False
    method: str = None
    threshold: Thresholding = Thresholding()
    small_object_removal_threshold: Thresholding = Thresholding()


@dataclass
class PostProcessConfig:
    zoom: Zoom = Zoom()
    thresholding: Thresholding = Thresholding()
    instance: InstanceSegConfig = InstanceSegConfig()


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
    """Class to record params for Inferer plugin"""

    model_info: ModelInfo = None
    show_results: bool = False
    show_results_count: int = 5
    show_original: bool = True
    anisotropy_resolution: List[int] = None


@dataclass
class InferenceWorkerConfig:
    """Class to record configuration for Inference job"""

    device: str
    model_info: ModelInfo
    weights_config: WeightsInfo
    results_path: str
    filetype: str
    keep_on_cpu: bool
    compute_stats: bool
    post_process_config: PostProcessConfig
    sliding_window_config: SlidingWindowConfig = SlidingWindowConfig()

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
    validation_interval: int = 2
    batch_size: int = 1
    results_path_folder: str = None
    sampling: bool = False
    num_samples: int = 2
    sample_size: List[int] = None
    do_augmentation: bool = True
    deterministic_config: DeterministicConfig = DeterministicConfig()
