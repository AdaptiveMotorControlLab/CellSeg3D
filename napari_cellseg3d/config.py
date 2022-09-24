import datetime
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List
from typing import Optional

import napari
import numpy as np

from napari_cellseg3d.model_instance_seg import binary_connected
from napari_cellseg3d.model_instance_seg import binary_watershed

# from napari_cellseg3d.models import model_TRAILMAP as TRAILMAP
from napari_cellseg3d.models import model_SegResNet as SegResNet
from napari_cellseg3d.models import model_SwinUNetR as SwinUNetR
from napari_cellseg3d.models import model_TRAILMAP_MS as TRAILMAP_MS
from napari_cellseg3d.models import model_VNet as VNet

# TODO DOCUMENT !!! and add default values
# TODO add JSON load/save

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

WEIGHTS_DIR = str(Path(__file__).parent.resolve() / Path("models/pretrained"))


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
    as_stack: bool = False
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
    """Dataclass recording model info :
    - name (str): name of the model"""

    name: str = next(iter(MODEL_LIST))
    model_input_size: Optional[List[int]] = None

    def get_model(self):
        try:
            return MODEL_LIST[self.name]
        except KeyError as e:
            warnings.warn(f"Model {self.name} is not defined")
            raise KeyError(e)

    @staticmethod
    def get_model_name_list():
        print(
            f"Model list :\n" + str(f"{name}\n" for name in MODEL_LIST.keys())
        )
        return MODEL_LIST.keys()


@dataclass
class WeightsInfo:
    path: str = WEIGHTS_DIR
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
    method: str = None
    threshold: Thresholding = Thresholding(enabled=False, threshold_value=0.85)
    small_object_removal_threshold: Thresholding = Thresholding(
        enabled=True, threshold_value=20
    )


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

    device: str = "cpu"
    model_info: ModelInfo = ModelInfo()
    weights_config: WeightsInfo = WeightsInfo()
    results_path: str = str(Path.home() / Path("cellseg3d/inference"))
    filetype: str = ".tif"
    keep_on_cpu: bool = False
    compute_stats: bool = False
    post_process_config: PostProcessConfig = PostProcessConfig()
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
    results_path_folder: str = str(Path.home() / Path("cellseg3d/training"))
    sampling: bool = False
    num_samples: int = 2
    sample_size: List[int] = None
    do_augmentation: bool = True
    deterministic_config: DeterministicConfig = DeterministicConfig()
