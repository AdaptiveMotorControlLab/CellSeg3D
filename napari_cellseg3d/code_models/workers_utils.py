"""Several worker-related utilities for inference and training."""
import typing as t
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from monai.transforms import MapTransform, Transform
from qtpy.QtCore import Signal
from superqt.utils._qthreading import WorkerBaseSignals
from tqdm import tqdm

# local
from napari_cellseg3d import interface as ui
from napari_cellseg3d import utils
from napari_cellseg3d.utils import LOGGER as logger

if TYPE_CHECKING:
    from napari_cellseg3d.code_models.instance_segmentation import ImageStats

PRETRAINED_WEIGHTS_DIR = Path(__file__).parent.resolve() / Path(
    "models/pretrained"
)


class WeightsDownloader:
    """A utility class the downloads the weights of a model when needed."""

    def __init__(self, log_widget: t.Optional[ui.Log] = None):
        """Creates a WeightsDownloader, optionally with a log widget to display the progress.

        Args:
            log_widget (log_utility.Log): a Log to display the progress bar in. If None, uses logger.info()
        """
        self.log_widget = log_widget

    def download_weights(self, model_name: str, model_weights_filename: str):
        """Downloads a specific pretrained model.

        This code is adapted from DeepLabCut with permission from MWMathis.

        Args:
            model_name (str): name of the model to download
            model_weights_filename (str): name of the .pth file expected for the model
        """
        import json
        import tarfile
        import urllib.request

        def show_progress(_, block_size, __):  # count, block_size, total_size
            pbar.update(block_size)

        logger.info("*" * 20)
        pretrained_folder_path = PRETRAINED_WEIGHTS_DIR
        json_path = pretrained_folder_path / Path("pretrained_model_urls.json")

        check_path = pretrained_folder_path / Path(model_weights_filename)

        if Path(check_path).is_file():
            message = f"Weight file {model_weights_filename} already exists, skipping download"
            if self.log_widget is not None:
                self.log_widget.print_and_log(message, printing=False)
            logger.info(message)
            return

        with Path.open(json_path) as f:
            neturls = json.load(f)
        if model_name in neturls:
            url = neturls[model_name]
            response = urllib.request.urlopen(url)

            start_message = f"Downloading the model from HuggingFace {url}...."
            total_size = int(response.getheader("Content-Length"))
            if self.log_widget is None:
                logger.info(start_message)
                pbar = tqdm(unit="B", total=total_size, position=0)
            else:
                self.log_widget.print_and_log(start_message)
                pbar = tqdm(
                    unit="B",
                    total=total_size,
                    position=0,
                    file=self.log_widget,
                )

            filename, _ = urllib.request.urlretrieve(
                url, reporthook=show_progress
            )
            with tarfile.open(filename, mode="r:gz") as tar:

                def is_within_directory(directory, target):
                    abs_directory = Path(directory).resolve()
                    abs_target = Path(target).resolve()
                    # prefix = os.path.commonprefix([abs_directory, abs_target])
                    logger.debug(abs_directory)
                    logger.debug(abs_target)
                    logger.debug(abs_directory.parents)

                    return abs_directory in abs_target.parents

                def safe_extract(
                    tar, path=".", members=None, *, numeric_owner=False
                ):
                    for member in tar.getmembers():
                        member_path = str(Path(path) / member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception(
                                "Attempted Path Traversal in Tar File"
                            )

                    tar.extractall(path, members, numeric_owner=numeric_owner)

                safe_extract(tar, pretrained_folder_path)
                # tar.extractall(pretrained_folder_path)
        else:
            raise ValueError(
                f"Unknown model: {model_name}. Should be one of {', '.join(neturls)}"
            )


class LogSignal(WorkerBaseSignals):
    """Signal to send messages to be logged from another thread.

    Separate from Worker instances as indicated `on this post`_

    .. _on this post: https://stackoverflow.com/questions/2970312/pyqt4-qtcore-pyqtsignal-object-has-no-attribute-connect
    """  # TODO link ?

    log_signal = Signal(str)
    """qtpy.QtCore.Signal: signal to be sent when some text should be logged"""
    log_w_replace_signal = Signal(str)
    """qtpy.QtCore.Signal: signal to be sent when some text should be logged, replacing the last line"""
    warn_signal = Signal(str)
    """qtpy.QtCore.Signal: signal to be sent when some warning should be emitted in main thread"""
    error_signal = Signal(Exception, str)
    """qtpy.QtCore.Signal: signal to be sent when some error should be emitted in main thread"""

    # Should not be an instance variable but a class variable, not defined in __init__, see
    # https://stackoverflow.com/questions/2970312/pyqt4-qtcore-pyqtsignal-object-has-no-attribute-connect

    def __init__(self, parent=None):
        """Creates a LogSignal."""
        super().__init__(parent=parent)


class TqdmToLogSignal:
    """File-like object to redirect tqdm output to the logger widget in the GUI that self.log emits to."""

    def __init__(self, log_func):
        """Creates a TqdmToLogSignal.

        Args:
        log_func (callable): function to call to log the output.
        """
        self.log_func = log_func

    def write(self, x):
        """Writes the output to the log_func."""
        self.log_func(x.strip())

    def flush(self):
        """Flushes the output. Unused."""
        pass


class ONNXModelWrapper(torch.nn.Module):
    """Class to replace torch model by ONNX Runtime session."""

    def __init__(self, file_location):
        """Creates an ONNXModelWrapper."""
        super().__init__()
        try:
            import onnxruntime as ort
        except ImportError as e:
            logger.error("ONNX is not installed but ONNX model was loaded")
            logger.error(e)
            msg = "PLEASE INSTALL ONNX CPU OR GPU USING: pip install napari-cellseg3d[onnx-cpu] OR pip install napari-cellseg3d[onnx-gpu]"
            logger.error(msg)
            raise ImportError(msg) from e

        self.ort_session = ort.InferenceSession(
            file_location,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

    def forward(self, modeL_input):
        """Wraps ONNX output in a torch tensor."""
        outputs = self.ort_session.run(
            None, {"input": modeL_input.cpu().numpy()}
        )
        return torch.tensor(outputs[0])

    def eval(self):
        """Dummy function.

        Replaces model.eval().
        """
        pass

    def to(self, device):
        """Dummy function.

        Replaces model.to(device).
        """
        pass


class QuantileNormalizationd(MapTransform):
    """MONAI-style dict transform to normalize each image in a batch individually by quantile normalization."""

    def __init__(self, keys, allow_missing_keys: bool = False):
        """Creates a QuantileNormalizationd transform."""
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        """Normalize each image in a batch individually by quantile normalization."""
        d = dict(data)
        for key in self.keys:
            d[key] = self.normalizer(d[key])
        return d

    def normalizer(self, image: torch.Tensor):
        """Normalize each image in a batch individually by quantile normalization."""
        if image.ndim == 4:
            for i in range(image.shape[0]):
                image[i] = utils.quantile_normalization(image[i])
        else:
            raise NotImplementedError(
                "QuantileNormalizationd only supports 2D and 3D tensors with NCHWD format"
            )
        return image


class QuantileNormalization(Transform):
    """MONAI-style transform to normalize each image in a batch individually by quantile normalization."""

    def __call__(self, img):
        """Normalize each image in a batch individually by quantile normalization."""
        return utils.quantile_normalization(img)


class RemapTensor(Transform):
    """Remap the values of a tensor to a new range."""

    def __init__(self, new_max, new_min):
        """Creates a RemapTensor transform.

        Args:
            new_max (float): new maximum value
            new_min (float): new minimum value
        """
        super().__init__()
        self.max = new_max
        self.min = new_min

    def __call__(self, img):
        """Remap the values of a tensor to a new range."""
        return utils.remap_image(img, new_max=self.max, new_min=self.min)


# class RemapTensord(MapTransform):
#     def __init__(
#         self, keys, new_max, new_min, allow_missing_keys: bool = False
#     ):
#         super().__init__(keys, allow_missing_keys)
#         self.max = new_max
#         self.min = new_min
#
#     def __call__(self, data):
#         d = dict(data)
#         for key in self.keys:
#             for i in range(d[key].shape[0]):
#                 logger.debug(f"remapping across channel {i}")
#                 d[key][i] = utils.remap_image(
#                     d[key][i], new_max=self.max, new_min=self.min
#                 )
#         return d


class Threshold(Transform):
    """Threshold a tensor to 0 or 1."""

    def __init__(self, threshold=0.5):
        """Creates a Threshold transform.

        Args:
            threshold (float): threshold value
        """
        super().__init__()
        self.threshold = threshold

    def __call__(self, img):
        """Threshold a tensor to 0 or 1."""
        res = torch.where(img > self.threshold, 1, 0)
        return torch.Tensor(res).float()


@dataclass
class InferenceResult:
    """Class to record results of a segmentation job."""

    image_id: int = 0
    original: np.array = None
    instance_labels: np.array = None
    crf_results: np.array = None
    stats: "np.array[ImageStats]" = None
    semantic_segmentation: np.array = None
    model_name: str = None


@dataclass
class TrainingReport:
    """Class to record results of a training job."""

    show_plot: bool = True
    epoch: int = 0
    loss_1_values: t.Dict = None  # example : {"Loss" : [0.1, 0.2, 0.3]}
    loss_2_values: t.List = None
    weights: np.array = None
    images_dict: t.Dict = (
        None  # output, discrete output, target, target labels
    )
    supervised: bool = True
    # OR decoder output, encoder output, target, target labels
    # format : {"Layer name" : {"data" : np.array, "cmap" : "turbo"}}
