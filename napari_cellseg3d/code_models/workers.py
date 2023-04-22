import platform
import time
import typing as t
from dataclasses import dataclass
from math import ceil
from pathlib import Path
import typing as t

import numpy as np
import torch

# MONAI
from monai.data import (
    CacheDataset,
    DataLoader,
    Dataset,
    PatchDataset,
    decollate_batch,
    pad_list_data_collate,
)
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    AddChannel,
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
    SpatialPad,
    SpatialPadd,
    ToTensor,
    Zoom,
)
from monai.utils import set_determinism

# from napari.qt.threading import thread_worker
# threads
from napari.qt.threading import GeneratorWorker

# from napari.qt.threading import thread_worker
from napari.qt.threading import WorkerBaseSignals

# Qt
from qtpy.QtCore import Signal
from tifffile import imwrite
from tqdm import tqdm

# local
from napari_cellseg3d import config, utils
from napari_cellseg3d import interface as ui
from napari_cellseg3d import utils

# local
from napari_cellseg3d.code_models.model_instance_seg import ImageStats
from napari_cellseg3d.code_models.model_instance_seg import volume_stats

logger = utils.LOGGER

"""
Writing something to log messages from outside the main thread is rather problematic (plenty of silent crashes...)
so instead, following the instructions in the guides below to have a worker with custom signals, I implemented
a custom worker function."""

# FutureReference():
# https://python-forum.io/thread-31349.html
# https://www.pythoncentral.io/pysidepyqt-tutorial-creating-your-own-signals-and-slots/
# https://napari-staging-site.github.io/guides/stable/threading.html

PRETRAINED_WEIGHTS_DIR = Path(__file__).parent.resolve() / Path(
    "models/pretrained"
)
VERBOSE_SCHEDULER = True
logger.debug(f"PRETRAINED WEIGHT DIR LOCATION : {PRETRAINED_WEIGHTS_DIR}")


class WeightsDownloader:
    """A utility class the downloads the weights of a model when needed."""

    def __init__(self, log_widget: t.Optional[ui.Log] = None):
        """
        Creates a WeightsDownloader, optionally with a log widget to display the progress.

        Args:
            log_widget (log_utility.Log): a Log to display the progress bar in. If None, uses logger.info()
        """
        self.log_widget = log_widget

    def download_weights(self, model_name: str, model_weights_filename: str):
        """
        Downloads a specific pretrained model.
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


"""
Writing something to log messages from outside the main thread needs specific care,
Following the instructions in the guides below to have a worker with custom signals,
a custom worker function was implemented.
"""

# https://python-forum.io/thread-31349.html
# https://www.pythoncentral.io/pysidepyqt-tutorial-creating-your-own-signals-and-slots/
# https://napari-staging-site.github.io/guides/stable/threading.html


class LogSignal(WorkerBaseSignals):
    """Signal to send messages to be logged from another thread.

    Separate from Worker instances as indicated `here`_

    .. _here: https://stackoverflow.com/questions/2970312/pyqt4-qtcore-pyqtsignal-object-has-no-attribute-connect
    """  # TODO link ?

    log_signal = Signal(str)
    """qtpy.QtCore.Signal: signal to be sent when some text should be logged"""
    warn_signal = Signal(str)
    """qtpy.QtCore.Signal: signal to be sent when some warning should be emitted in main thread"""
    error_signal = Signal(Exception, str)
    """qtpy.QtCore.Signal: signal to be sent when some error should be emitted in main thread"""

    # Should not be an instance variable but a class variable, not defined in __init__, see
    # https://stackoverflow.com/questions/2970312/pyqt4-qtcore-pyqtsignal-object-has-no-attribute-connect

    def __init__(self):
        super().__init__()


# TODO(cyril): move inference and training workers to separate files


class ONNXModelWrapper(torch.nn.Module):
    """Class to replace torch model by ONNX Runtime session"""

    def __init__(self, file_location):
        super().__init__()
        try:
            import onnxruntime as ort
        except ImportError as e:
            logger.error("ONNX is not installed but ONNX model was loaded")
            logger.error(e)
            msg = "PLEASE INSTALL ONNX CPU OR GPU USING pip install napari-cellseg3d[onnx-cpu] OR napari-cellseg3d[onnx-gpu]"
            logger.error(msg)
            raise ImportError(msg) from e

        self.ort_session = ort.InferenceSession(
            file_location,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

    def forward(self, modeL_input):
        """Wraps ONNX output in a torch tensor"""
        outputs = self.ort_session.run(
            None, {"input": modeL_input.cpu().numpy()}
        )
        return torch.tensor(outputs[0])

    def eval(self):
        """Dummy function to replace model.eval()"""
        pass

    def to(self, device):
        """Dummy function to replace model.to(device)"""
        pass


@dataclass
class InferenceResult:
    """Class to record results of a segmentation job"""

    image_id: int = 0
    original: np.array = None
    instance_labels: np.array = None
    stats: "np.array[ImageStats]" = None
    result: np.array = None
    model_name: str = None


class InferenceWorker(GeneratorWorker):
    """A custom worker to run inference jobs in.
    Inherits from :py:class:`napari.qt.threading.GeneratorWorker`"""

    def __init__(
        self,
        worker_config: config.InferenceWorkerConfig,
    ):
        """Initializes a worker for inference with the arguments needed by the :py:func:`~inference` function.

        The config contains the following attributes:
        * device: cuda or cpu device to use for torch
        * model_dict: the :py:attr:`~self.models_dict` dictionary to obtain the model name, class and instance
        * weights_dict: dict with "custom" : bool to use custom weights or not; "path" : the path to weights if custom or name of the file if not custom
        * results_path: the path to save the results to
        * filetype: the file extension to use when saving,
        * transforms: a dict containing transforms to perform at various times.
        * instance: a dict containing parameters regarding instance segmentation
        * use_window: use window inference with specific size or whole image
        * window_infer_size: size of window if use_window is True
        * keep_on_cpu: keep images on CPU or no
        * stats_csv: compute stats on cells and save them to a csv file
        * images_filepaths: the paths to the images of the dataset
        * layer: the layer to run inference on

        Args:
            * worker_config (config.InferenceWorkerConfig): dataclass containing the proper configuration elements

        Note: See :py:func:`~self.inference`
        """

        super().__init__(self.inference)
        self._signals = LogSignal()  # add custom signals
        self.log_signal = self._signals.log_signal
        self.warn_signal = self._signals.warn_signal
        self.error_signal = self._signals.error_signal

        self.config = worker_config

        """These attributes are all arguments of :py:func:~inference, please see that for reference"""

        self.downloader = WeightsDownloader()
        """Download utility"""

    @staticmethod
    def create_inference_dict(images_filepaths):
        """Create a dict for MONAI with "image" keys with all image paths in :py:attr:`~self.images_filepaths`

        Returns:
            dict: list of image paths from loaded folder"""
        return [{"image": image_name} for image_name in images_filepaths]

    def set_download_log(self, widget):
        self.downloader.log_widget = widget

    def log(self, text):
        """Sends a signal that ``text`` should be logged

        Args:
            text (str): text to logged
        """
        self.log_signal.emit(text)

    def warn(self, warning):
        """Sends a warning to main thread"""
        self.warn_signal.emit(warning)

    def raise_error(self, exception, msg):
        """Raises an error in main thread"""
        logger.error(msg, exc_info=True)
        logger.error(exception, exc_info=True)

        self.log_signal.emit("!" * 20)
        self.log_signal.emit("Error occured")
        # self.log_signal.emit(msg)
        # self.log_signal.emit(str(exception))

        self.error_signal.emit(exception, msg)
        self.errored.emit(exception)
        yield exception
        # self.quit()

    def log_parameters(self):
        config = self.config

        self.log("-" * 20)
        self.log("\nParameters summary :")

        self.log(f"Model is : {config.model_info.name}")
        if config.post_process_config.thresholding.enabled:
            self.log(
                f"Thresholding is enabled at {config.post_process_config.thresholding.threshold_value}"
            )

        status = (
            "enabled"
            if config.sliding_window_config.is_enabled()
            else "disabled"
        )

        self.log(f"Window inference is {status}\n")
        if status == "enabled":
            self.log(
                f"Window size is {self.config.sliding_window_config.window_size}"
            )
            self.log(
                f"Window overlap is {self.config.sliding_window_config.window_overlap}"
            )

        if config.keep_on_cpu:
            self.log("Dataset loaded to CPU")
        else:
            self.log(f"Dataset loaded on {config.device}")

        if config.post_process_config.zoom.enabled:
            self.log(
                f"Scaling factor : {config.post_process_config.zoom.zoom_values} (x,y,z)"
            )

        instance_config = config.post_process_config.instance
        if instance_config.enabled:
            self.log(
                f"Instance segmentation enabled, method : {instance_config.method.name}\n"
            )
        self.log("-" * 20)

    def load_folder(self):
        images_dict = self.create_inference_dict(self.config.images_filepaths)

        # TODO : better solution than loading first image always ?
        data_check = LoadImaged(keys=["image"])(images_dict[0])

        check = data_check["image"].shape

        # self.log("\nChecking dimensions...")
        pad = utils.get_padding_dim(check)

        # dims = self.model_dict["model_input_size"]
        #
        # if self.model_dict["name"] == "SegResNet":
        #     model = self.model_dict["class"].get_net(
        #         input_image_size=[
        #             dims,
        #             dims,
        #             dims,
        #         ]
        #     )
        # elif self.model_dict["name"] == "SwinUNetR":
        #     model = self.model_dict["class"].get_net(
        #         img_size=[dims, dims, dims],
        #         use_checkpoint=False,
        #     )
        # else:
        #     model = self.model_dict["class"].get_net()
        #
        # self.log_parameters()
        #
        # model.to(self.config.device)

        # logger.debug("FILEPATHS PRINT")
        # logger.debug(self.images_filepaths)
        if self.config.sliding_window_config.is_enabled():
            load_transforms = Compose(
                [
                    LoadImaged(keys=["image"]),
                    # AddChanneld(keys=["image"]), #already done
                    EnsureChannelFirstd(keys=["image"]),
                    # Orientationd(keys=["image"], axcodes="PLI"),
                    # anisotropic_transform,
                    EnsureTyped(keys=["image"]),
                ]
            )
        else:
            load_transforms = Compose(
                [
                    LoadImaged(keys=["image"]),
                    # AddChanneld(keys=["image"]), #already done
                    EnsureChannelFirstd(keys=["image"]),
                    # Orientationd(keys=["image"], axcodes="PLI"),
                    # anisotropic_transform,
                    SpatialPadd(keys=["image"], spatial_size=pad),
                    EnsureTyped(keys=["image"]),
                ]
            )

        self.log("\nLoading dataset...")
        inference_ds = Dataset(data=images_dict, transform=load_transforms)
        inference_loader = DataLoader(
            inference_ds, batch_size=1, num_workers=2
        )
        self.log("Done")
        return inference_loader

    def load_layer(self):
        self.log("\nLoading layer\n")
        data = np.squeeze(self.config.layer)

        volume = np.array(data, dtype=np.int16)

        volume_dims = len(volume.shape)
        if volume_dims != 3:
            raise ValueError(
                f"Data array is not 3-dimensional but {volume_dims}-dimensional,"
                f" please check for extra channel/batch dimensions"
            )

        volume = np.swapaxes(
            volume, 0, 2
        )  # for anisotropy to be monai-like, i.e. zyx # FIXME rotation not always correct

        dims_check = volume.shape
        self.log("Checking dimensions...")
        pad = utils.get_padding_dim(dims_check)

        # logger.debug(volume.shape)
        # logger.debug(volume.dtype)
        if self.config.sliding_window_config.is_enabled():
            load_transforms = Compose(
                [
                    ToTensor(),
                    # anisotropic_transform,
                    AddChannel(),
                    # SpatialPad(spatial_size=pad),
                    AddChannel(),
                    EnsureType(),
                ],
                map_items=False,
                log_stats=True,
            )
        else:
            load_transforms = Compose(
                [
                    ToTensor(),
                    # anisotropic_transform,
                    AddChannel(),
                    SpatialPad(spatial_size=pad),
                    AddChannel(),
                    EnsureType(),
                ],
                map_items=False,
                log_stats=True,
            )

        input_image = load_transforms(volume)
        self.log("Done")
        return input_image

    def model_output(
        self,
        inputs,
        model,
        post_process_transforms,
        post_process=True,
        aniso_transform=None,
    ):
        inputs = inputs.to("cpu")

        # def model_output(inputs):
        #     return post_process_transforms(
        #         self.config.model_info.get_model().get_output(model, inputs)
        #     )

        if self.config.keep_on_cpu:
            dataset_device = "cpu"
        else:
            dataset_device = self.config.device

        if self.config.sliding_window_config.is_enabled():
            window_size = self.config.sliding_window_config.window_size
            window_size = [window_size, window_size, window_size]
            window_overlap = self.config.sliding_window_config.window_overlap
        else:
            window_size = None
            window_overlap = 0
        try:
            # logger.debug(f"model : {model}")
            logger.debug(f"inputs shape : {inputs.shape}")
            logger.debug(f"inputs type : {inputs.dtype}")
            try:
                # outputs = model(inputs)

                def model_output_wrapper(inputs):
                    result = model(inputs)
                    return post_process_transforms(result)

                with torch.no_grad():
                    outputs = sliding_window_inference(
                        inputs,
                        roi_size=window_size,
                        sw_batch_size=1,  # TODO add param
                        predictor=model_output_wrapper,
                        sw_device=self.config.device,
                        device=dataset_device,
                        overlap=window_overlap,
                        progress=True,
                    )
            except Exception as e:
                logger.error(e, exc_info=True)
                logger.debug("failed to run sliding window inference")
                self.raise_error(e, "Error during sliding window inference")
            logger.debug(f"Inference output shape: {outputs.shape}")
            self.log("Post-processing...")
            out = outputs.detach().cpu().numpy()
            if aniso_transform is not None:
                out = aniso_transform(out)
            if post_process:
                out = np.array(out).astype(np.float32)
                out = np.squeeze(out)
                return out
            else:
                return out
        except Exception as e:
            logger.error(e, exc_info=True)
            self.raise_error(e, "Error during sliding window inference")
            # sys.stdout = old_stdout
            # sys.stderr = old_stderr

    def create_inference_result(
        self,
        semantic_labels,
        instance_labels,
        crf_results=None,
        from_layer: bool = False,
        original=None,
        stats=None,
        i=0,
    ):
        if not from_layer and original is None:
            raise ValueError(
                "If the image is not from a layer, an original should always be available"
            )

        if from_layer:
            if i != 0:
                raise ValueError(
                    "A layer's ID should always be 0 (default value)"
                )
            total_dims = len(semantic_labels.shape) - 3
            semantic_labels = np.swapaxes(
                semantic_labels, 0 + total_dims, 2 + total_dims
            )

        return InferenceResult(
            image_id=i + 1,
            original=original,
            instance_labels=instance_labels,
            crf_results=crf_results,
            stats=stats,
            result=semantic_labels,
            model_name=self.config.model_info.name,
        )

    def get_original_filename(self, i):
        return Path(self.config.images_filepaths[i]).stem

    def get_instance_result(self, semantic_labels, from_layer=False, i=-1):
        if not from_layer and i == -1:
            raise ValueError(
                "An ID should be provided when running from a file"
            )

        if self.config.post_process_config.instance.enabled:
            instance_labels = self.instance_seg(
                semantic_labels,
                i + 1,
            )
            if from_layer:
                instance_labels = np.swapaxes(
                    instance_labels, 0, 2
                )  # TODO(cyril) check if correct
            data_dict = self.stats_csv(instance_labels)
        else:
            instance_labels = None
            data_dict = None
        return instance_labels, data_dict

    def save_image(
        self,
        image,
        from_layer=False,
        i=0,
        additional_info="",
    ):
        if not from_layer:
            original_filename = "_" + self.get_original_filename(i) + "_"
            filetype = self.config.filetype
        else:
            original_filename = "_"
            filetype = ""

        time = utils.get_date_time()

        file_path = (
            self.config.results_path
            + "/"
            + f"{additional_info}"
            + f"Prediction_{i+1}"
            + original_filename
            + self.config.model_info.name
            + f"_{time}_"
            + filetype
        )
        try:
            imwrite(file_path, image)
        except ValueError as e:
            self.raise_error(e, "Error during image saving")
        filename = Path(file_path).stem

        if from_layer:
            self.log(f"\nLayer prediction saved as : {filename}")
        else:
            self.log(f"\nFile n°{i+1} saved as : {filename}")

    def aniso_transform(self, image):
        if self.config.post_process_config.zoom.enabled:
            zoom = self.config.post_process_config.zoom.zoom_values
            anisotropic_transform = Zoom(
                zoom=zoom,
                keep_size=False,
                padding_mode="empty",
            )
            return anisotropic_transform(image[0])
        else:
            return image

    def instance_seg(
        self, to_instance, image_id=0, original_filename="layer", channel=None
    ):
        if image_id is not None:
            self.log(f"\nRunning instance segmentation for image n°{image_id}")

        method = self.config.post_process_config.instance.method
        instance_labels = method.run_method(image=to_instance)

        if channel is not None:
            channel_id = f"_{channel}"
        else:
            channel_id = ""

        if self.config.filetype == "":
            filetype = ""
        else:
            filetype = "_" + self.config.filetype

        instance_filepath = (
            self.config.results_path
            + "/"
            + f"Instance_seg_labels_{image_id}_"
            + original_filename
            + channel_id
            + "_"
            + self.config.model_info.name
            + f"_{utils.get_date_time()}"
            + filetype
        )

        imwrite(instance_filepath, instance_labels)
        self.log(
            f"Instance segmentation results for image n°{image_id} have been saved as:"
        )
        self.log(Path(instance_filepath).name)
        return instance_labels

    def inference_on_folder(self, inf_data, i, model, post_process_transforms):
        self.log("-" * 10)
        self.log(f"Inference started on image n°{i + 1}...")

        inputs = inf_data["image"]

        out = self.model_output(
            inputs,
            model,
            post_process_transforms,
            aniso_transform=self.aniso_transform,
        )

        self.save_image(out, i=i)
        instance_labels, stats = self.get_instance_result(out, i=i)
        if self.config.use_crf:
            try:
                crf_results = self.run_crf(inputs, out, image_id=i)

            except ValueError as e:
                self.log(f"Error occurred during CRF : {e}")
                crf_results = None
        else:
            crf_results = None

        original = np.array(inf_data["image"]).astype(np.float32)

        self.log(f"Inference completed on image n°{i+1}")

        return self.create_inference_result(
            out,
            instance_labels,
            crf_results,
            from_layer=False,
            original=original,
            stats=stats,
            i=i,
        )

    def run_crf(self, image, labels, image_id=0):
        try:
            crf_results = crf_with_config(
                image, labels, config=self.config.crf_config, log=self.log
            )
            self.save_image(
                crf_results,
                i=image_id,
                additional_info="CRF_",
                from_layer=True,
            )
            return crf_results
        except ValueError as e:
            self.log(f"Error occurred during CRF : {e}")
            return None

    def stats_csv(self, instance_labels):
        if self.config.compute_stats:
            stats = volume_stats(instance_labels)
            return stats
        except ValueError as e:
            self.log(f"Error occurred during stats computing : {e}")
            return None

    def inference_on_layer(self, image, model, post_process_transforms):
        self.log("-" * 10)
        self.log("Inference started on layer...")

        image = image.type(torch.FloatTensor)

        out = self.model_output(
            image,
            model,
            post_process_transforms,
            aniso_transform=self.aniso_transform,
        )

        self.save_image(out, from_layer=True)

        instance_labels_results = []
        stats_results = []

        for channel in out:
            instance_labels, stats = self.get_instance_result(
                channel, from_layer=True
            )
            instance_labels_results.append(instance_labels)
            stats_results.append(stats)

        return self.create_inference_result(
            semantic_labels=out,
            instance_labels=instance_labels_results,
            from_layer=True,
            stats=stats_results,
        )

    # @thread_worker(connect={"errored": self.raise_error})
    def inference(self):
        """
        Requires:
            * device: cuda or cpu device to use for torch

            * model_dict: the :py:attr:`~self.models_dict` dictionary to obtain the model name, class and instance

            * weights: the loaded weights from the model

            * images_filepaths: the paths to the images of the dataset

            * results_path: the path to save the results to

            * filetype: the file extension to use when saving,

            * transforms: a dict containing transforms to perform at various times.

            * use_window: use window inference with specific size or whole image

            * window_infer_size: size of window if use_window is True

            * keep_on_cpu: keep images on CPU or no

            * stats_csv: compute stats on cells and save them to a csv file

        Yields:
            dict: contains :
                * "image_id" : index of the returned image

                * "original" : original volume used for inference

                * "result" : inference result

        """
        sys = platform.system()
        logger.debug(f"OS is {sys}")
        if sys == "Darwin":
            torch.set_num_threads(1)  # required for threading on macOS ?
            self.log("Number of threads has been set to 1 for macOS")

        try:
            dims = self.config.model_info.model_input_size
            self.log(f"MODEL DIMS : {dims}")
            model_name = self.config.model_info.name
            model_class = self.config.model_info.get_model()
            self.log(f"Model name : {model_name}")

            weights_config = self.config.weights_config
            post_process_config = self.config.post_process_config

            # try:
            self.log("Instantiating model...")
            model = model_class(  # FIXME test if works
                input_img_size=[dims, dims, dims],
                device=self.config.device,
                num_classes=self.config.model_info.num_classes,
            )
            # try:
            model = model.to(self.config.device)
            # except Exception as e:
            #     self.raise_error(e, "Issue loading model to device")
            # logger.debug(f"model : {model}")
            if model is None:
                raise ValueError("Model is None")
            # try:
            self.log("\nLoading weights...")
            if weights_config.custom:
                weights = weights_config.path
            else:
                self.downloader.download_weights(
                    model_name,
                    model_class.weights_file,
                )
                weights = str(
                    PRETRAINED_WEIGHTS_DIR / Path(model_class.weights_file)
                )
            model.load_state_dict(
                torch.load(
                    weights,
                    map_location=self.config.device,
                )
            )
            self.log("Done")
            # except Exception as e:
            #     self.raise_error(e, "Issue loading weights")
            # except Exception as e:
            #     self.raise_error(e, "Issue instantiating model")

            # if model_name == "SegResNet":
            #     model = model_class(
            #         input_image_size=[
            #             dims,
            #             dims,
            #             dims,
            #         ],
            #     )
            # elif model_name == "SwinUNetR":
            #     model = model_class(
            #         img_size=[dims, dims, dims],
            #         use_checkpoint=False,
            #     )
            # else:
            #     model = model_class.get_net()

            self.log_parameters()

            # load_transforms = Compose(
            #     [
            #         LoadImaged(keys=["image"]),
            #         # AddChanneld(keys=["image"]), #already done
            #         EnsureChannelFirstd(keys=["image"]),
            #         # Orientationd(keys=["image"], axcodes="PLI"),
            #         # anisotropic_transform,
            #         SpatialPadd(keys=["image"], spatial_size=pad),
            #         EnsureTyped(keys=["image"]),
            #     ]
            # )

            if not post_process_config.thresholding.enabled:
                post_process_transforms = EnsureType()
            else:
                t = post_process_config.thresholding.threshold_value
                post_process_transforms = Compose(
                    AsDiscrete(threshold=t), EnsureType()
                )

            is_folder = self.config.images_filepaths is not None
            is_layer = self.config.layer is not None

            if is_layer and is_folder:
                raise ValueError(
                    "Both a layer and a folder have been specified, please specify only one of the two. Aborting."
                )
            elif is_folder:
                inference_loader = self.load_folder()
                ##################
                ##################
                # DEBUG
                # from monai.utils import first
                #
                # check_data = first(inference_loader)
                # image = check_data[0][0]
                # logger.debug(image.shape)
                ##################
                ##################
            elif is_layer:
                input_image = self.load_layer()
            else:
                raise ValueError("No data has been provided. Aborting.")

            if model is None:
                raise ValueError("Model is None")

            model.eval()
            with torch.no_grad():
                ################################
                ################################
                ################################
                if is_folder:
                    for i, inf_data in enumerate(inference_loader):
                        yield self.inference_on_folder(
                            inf_data, i, model, post_process_transforms
                        )
                elif is_layer:
                    yield self.inference_on_layer(
                        input_image, model, post_process_transforms
                    )
            model.to("cpu")
            # self.quit()
        except Exception as e:
            logger.error(e, exc_info=True)
            self.raise_error(e, "Inference failed")
            self.quit()
        finally:
            self.quit()


@dataclass
class TrainingReport:
    show_plot: bool = True
    epoch: int = 0
    loss_values: t.Dict = None  # TODO(cyril) : change to dict and unpack different losses for e.g. WNet with several losses
    validation_metric: t.List = None
    weights: np.array = None
    images: t.List[np.array] = None


class TrainingWorker(GeneratorWorker):
    """A custom worker to run training jobs in.
    Inherits from :py:class:`napari.qt.threading.GeneratorWorker`"""

    def __init__(
        self,
        worker_config: config.TrainingWorkerConfig,
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
        super().__init__(self.train)
        self._signals = LogSignal()
        self.log_signal = self._signals.log_signal
        self.warn_signal = self._signals.warn_signal
        self.error_signal = self._signals.error_signal

        self._weight_error = False
        #############################################
        self.config = worker_config

        self.train_files = []
        self.val_files = []
        #######################################
        self.downloader = WeightsDownloader()

    def set_download_log(self, widget):
        self.downloader.log_widget = widget

    def log(self, text):
        """Sends a signal that ``text`` should be logged

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

            if do_sampling:
                size = self.config.sample_size
            else:
                size = check

            model = model_class(  # FIXME check if correct
                input_img_size=utils.get_padding_dim(size), use_checkpoint=True
            )
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
                        SpatialPadd(
                            keys=["image", "label"],
                            spatial_size=(utils.get_padding_dim(check)),
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
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                mode="min",
                factor=self.config.scheduler_factor,
                patience=self.config.scheduler_patience,
                verbose=VERBOSE_SCHEDULER,
            )
            dice_metric = DiceMetric(include_background=True, reduction="mean")

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
                        )
                    )
                except RuntimeError as e:
                    logger.error(f"Error when loading weights : {e}")
                    logger.error(e, exc_info=True)
                    warn = (
                        "WARNING:\nIt'd seem that the weights were incompatible with the model,\n"
                        "the model will be trained from random weights"
                    )
                    self.log(warn)
                    self.warn(warn)
                    self._weight_error = True

            if self.config.device.type == "cuda":
                self.log("\nUsing GPU :")
                self.log(torch.cuda.get_device_name(0))
            else:
                self.log("Using CPU")

            self.log_parameters()

            device = self.config.device

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
                        torch.cuda.memory_allocated(0) / 1024**3, 1
                    )
                    reserved_mem = round(
                        torch.cuda.memory_reserved(0) / 1024**3, 1
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
                        ]  # FIXME fix channel number
                        if len(outputs.shape) < 4:
                            outputs = outputs.unsqueeze(0)
                    loss = self.config.loss_function(outputs, labels)
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

                epoch_loss /= step
                epoch_loss_values.append(epoch_loss)
                self.log(f"Epoch: {epoch + 1}, Average loss: {epoch_loss:.4f}")

                self.log("Updating scheduler...")
                scheduler.step(epoch_loss)

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
                            self.log("Performing validation...")
                            try:
                                with torch.no_grad():
                                    val_outputs = sliding_window_inference(
                                        val_inputs,
                                        roi_size=size,
                                        sw_batch_size=self.config.batch_size,
                                        predictor=model,
                                        overlap=0.25,
                                        sw_device=self.config.device,
                                        device=self.config.device,
                                        progress=True,
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
                                AsDiscrete(threshold=0.6), EnsureType()
                            )  #
                            post_label = EnsureType()

                            val_outputs = [
                                post_pred(res_tensor) for res_tensor in pred
                            ]

                            val_labels = [
                                post_label(res_tensor) for res_tensor in labs
                            ]

                            # logger.debug(len(val_outputs))
                            # logger.debug(len(val_labels))

                            dice_metric(y_pred=val_outputs, y=val_labels)
                            checkpoint_output.append(
                                [res.detach().cpu() for res in val_outputs]
                            )

                        checkpoint_output = [
                            item.numpy()
                            for batch in checkpoint_output
                            for item in batch
                        ]

                        metric = dice_metric.aggregate().detach().item()
                        dice_metric.reset()

                        val_metric_values.append(metric)

                        train_report = TrainingReport(
                            show_plot=True,
                            epoch=epoch,
                            loss_values=epoch_loss_values,
                            validation_metric=val_metric_values,
                            weights=model.state_dict(),
                            images=checkpoint_output,
                        )
                        self.log("Validation completed")
                        yield train_report

                        weights_filename = (
                            f"{model_name}_best_metric"
                            + f"_epoch_{epoch + 1}.pth"
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
            model.to("cpu")

        except Exception as e:
            self.raise_error(e, "Error in training")
            self.quit()
        finally:
            self.quit()
