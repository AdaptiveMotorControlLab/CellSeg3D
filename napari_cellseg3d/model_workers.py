import os
import platform
from pathlib import Path
import importlib.util
from typing import Optional

import numpy as np
from tifffile import imwrite
import torch
from tqdm import tqdm

# MONAI
from monai.data import CacheDataset
from monai.data import DataLoader
from monai.data import Dataset
from monai.data import decollate_batch
from monai.data import pad_list_data_collate
from monai.data import PatchDataset
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import AddChannel
from monai.transforms import AsDiscrete
from monai.transforms import Compose
from monai.transforms import EnsureChannelFirstd
from monai.transforms import EnsureType
from monai.transforms import EnsureTyped
from monai.transforms import LoadImaged
from monai.transforms import Orientationd
from monai.transforms import Rand3DElasticd
from monai.transforms import RandAffined
from monai.transforms import RandFlipd
from monai.transforms import RandRotate90d
from monai.transforms import RandShiftIntensityd
from monai.transforms import RandSpatialCropSamplesd
from monai.transforms import SpatialPad
from monai.transforms import SpatialPadd
from monai.transforms import ToTensor
from monai.transforms import Zoom
from monai.utils import set_determinism

# threads
from napari.qt.threading import GeneratorWorker
from napari.qt.threading import WorkerBaseSignals

# Qt
from qtpy.QtCore import Signal

from napari_cellseg3d import utils
from napari_cellseg3d import log_utility

# local
from napari_cellseg3d.model_instance_seg import binary_connected
from napari_cellseg3d.model_instance_seg import binary_watershed
from napari_cellseg3d.model_instance_seg import volume_stats

"""
Writing something to log messages from outside the main thread is rather problematic (plenty of silent crashes...)
so instead, following the instructions in the guides below to have a worker with custom signals, I implemented
a custom worker function."""

# FutureReference():
# https://python-forum.io/thread-31349.html
# https://www.pythoncentral.io/pysidepyqt-tutorial-creating-your-own-signals-and-slots/
# https://napari-staging-site.github.io/guides/stable/threading.html

WEIGHTS_DIR = os.path.dirname(os.path.realpath(__file__)) + str(
    Path("/models/pretrained")
)


class WeightsDownloader:
    """A utility class the downloads the weights of a model when needed."""

    def __init__(self, log_widget: Optional[log_utility.Log] = None):
        """
        Creates a WeightsDownloader, optionally with a log widget to display the progress.

        Args:
            log_widget (log_utility.Log): a Log to display the progress bar in. If None, uses print()
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

        def show_progress(count, block_size, total_size):
            pbar.update(block_size)

        cellseg3d_path = os.path.split(
            importlib.util.find_spec("napari_cellseg3d").origin
        )[0]
        pretrained_folder_path = os.path.join(
            cellseg3d_path, "models", "pretrained"
        )
        json_path = os.path.join(
            pretrained_folder_path, "pretrained_model_urls.json"
        )

        check_path = os.path.join(
            pretrained_folder_path, model_weights_filename
        )
        if os.path.exists(check_path):
            message = f"Weight file {model_weights_filename} already exists, skipping download"
            if self.log_widget is not None:
                self.log_widget.print_and_log(message, printing=False)
            print(message)
            return

        with open(json_path) as f:
            neturls = json.load(f)
        if model_name in neturls.keys():
            url = neturls[model_name]
            response = urllib.request.urlopen(url)

            start_message = f"Downloading the model from the M.W. Mathis Lab server {url}...."
            total_size = int(response.getheader("Content-Length"))
            if self.log_widget is None:
                print(start_message)
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
                tar.extractall(pretrained_folder_path)
        else:
            raise ValueError(
                f"Unknown model: {model_name}. Should be one of {', '.join(neturls)}"
            )


class LogSignal(WorkerBaseSignals):
    """Signal to send messages to be logged from another thread.

    Separate from Worker instances as indicated `here`_"""  # TODO link ?

    log_signal = Signal(str)
    """qtpy.QtCore.Signal: signal to be sent when some text should be logged"""
    warn_signal = Signal(str)
    """qtpy.QtCore.Signal: signal to be sent when some warning should be emitted in main thread"""

    # Should not be an instance variable but a class variable, not defined in __init__, see
    # https://stackoverflow.com/questions/2970312/pyqt4-qtcore-pyqtsignal-object-has-no-attribute-connect

    def __init__(self):
        super().__init__()


# TODO : use dataclass for config instead ?


class InferenceWorker(GeneratorWorker):
    """A custom worker to run inference jobs in.
    Inherits from :py:class:`napari.qt.threading.GeneratorWorker`"""

    def __init__(
        self,
        device,
        model_dict,
        weights_dict,
        results_path,
        filetype,
        transforms,
        instance,
        use_window,
        window_infer_size,
        window_overlap,
        keep_on_cpu,
        stats_csv,
        images_filepaths=None,
        layer=None,  # FIXME
    ):
        """Initializes a worker for inference with the arguments needed by the :py:func:`~inference` function.

        Args:
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
        Note: See :py:func:`~self.inference`
        """

        super().__init__(self.inference)
        self._signals = LogSignal()  # add custom signals
        self.log_signal = self._signals.log_signal
        self.warn_signal = self._signals.warn_signal
        ###########################################
        ###########################################
        self.device = device
        self.model_dict = model_dict
        self.weights_dict = weights_dict
        self.results_path = results_path
        self.filetype = filetype
        self.transforms = transforms
        self.instance_params = instance
        self.use_window = use_window
        self.window_infer_size = window_infer_size
        self.window_overlap_percentage = window_overlap
        self.keep_on_cpu = keep_on_cpu
        self.stats_to_csv = stats_csv
        ############################################
        ############################################
        self.layer = layer
        self.images_filepaths = images_filepaths
        ############################################
        ############################################

        """These attributes are all arguments of :py:func:~inference, please see that for reference"""

        self.downloader = WeightsDownloader()
        """Download utility"""

    @staticmethod
    def create_inference_dict(images_filepaths):
        """Create a dict for MONAI with "image" keys with all image paths in :py:attr:`~self.images_filepaths`

        Returns:
            dict: list of image paths from loaded folder"""
        data_dicts = [{"image": image_name} for image_name in images_filepaths]
        return data_dicts

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

    def log_parameters(self):

        self.log("-" * 20)
        self.log("\nParameters summary :")

        self.log(f"Model is : {self.model_dict['name']}")
        if self.transforms["thresh"][0]:
            self.log(
                f"Thresholding is enabled at {self.transforms['thresh'][1]}"
            )

        if self.use_window:
            status = "enabled"
        else:
            status = "disabled"

        self.log(f"Window inference is {status}\n")

        if self.keep_on_cpu:
            self.log(f"Dataset loaded to CPU")
        else:
            self.log(f"Dataset loaded on {self.device}")

        if self.transforms["zoom"][0]:
            self.log(f"Scaling factor : {self.transforms['zoom'][1]} (x,y,z)")

        if self.instance_params["do_instance"]:
            self.log(
                f"Instance segmentation enabled, method : {self.instance_params['method']}\n"
                f"Probability threshold is {self.instance_params['threshold']:.2f}\n"
                f"Objects smaller than {self.instance_params['size_small']} pixels will be removed\n"
            )
        self.log("-" * 20)

    def load_folder(self):

        images_dict = self.create_inference_dict(self.images_filepaths)

        # TODO : better solution than loading first image always ?
        data_check = LoadImaged(keys=["image"])(images_dict[0])

        check = data_check["image"].shape

        self.log("\nChecking dimensions...")
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
        # model.to(self.device)

        # print("FILEPATHS PRINT")
        # print(self.images_filepaths)
        if self.use_window:
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

        data = np.squeeze(self.layer.data)

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
        print("Loading layer\n")
        dims_check = volume.shape
        self.log("\nChecking dimensions...")
        pad = utils.get_padding_dim(dims_check)

        # print(volume.shape)
        # print(volume.dtype)
        if self.use_window:
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

        self.log("\nLoading dataset...")
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

        model_output = lambda inputs: post_process_transforms(
            self.model_dict["class"].get_output(model, inputs)
        )

        if self.keep_on_cpu:
            dataset_device = "cpu"
        else:
            dataset_device = self.device

        if self.use_window:
            window_size = self.window_infer_size
            window_overlap = self.window_overlap_percentage
        else:
            window_size = None
            window_overlap = 0.25

        outputs = sliding_window_inference(
            inputs,
            roi_size=window_size,
            sw_batch_size=1,
            predictor=model_output,
            sw_device=self.device,
            device=dataset_device,
            overlap=window_overlap,
        )

        out = outputs.detach().cpu()

        if aniso_transform is not None:
            out = aniso_transform(out)

        if post_process:
            out = np.array(out).astype(np.float32)
            out = np.squeeze(out)
            return out
        else:
            return out

    def create_result_dict(
        self,
        semantic_labels,
        instance_labels,
        from_layer: bool,
        original=None,
        data_dict=None,
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
            semantic_labels = np.swapaxes(semantic_labels, 0, 2)

        return {
            "image_id": i + 1,
            "original": original,
            "instance_labels": instance_labels,
            "object stats": data_dict,
            "result": semantic_labels,
            "model_name": self.model_dict["name"],
        }

    def get_original_filename(self, i):
        return os.path.basename(self.images_filepaths[i]).split(".")[0]

    def get_instance_result(self, semantic_labels, from_layer=False, i=-1):

        if not from_layer and i == -1:
            raise ValueError(
                "An ID should be provided when running from a file"
            )

        if self.instance_params["do_instance"]:
            instance_labels = self.instance_seg(
                semantic_labels,
                i + 1,
            )
            if from_layer:
                instance_labels = np.swapaxes(instance_labels, 0, 2)
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
    ):

        if not from_layer:
            original_filename = "_" + self.get_original_filename(i) + "_"
        else:
            original_filename = "_"

        time = utils.get_date_time()

        file_path = (
            self.results_path
            + "/"
            + f"Prediction_{i+1}"
            + original_filename
            + self.model_dict["name"]
            + f"_{time}_"
            + self.filetype
        )

        imwrite(file_path, image)
        filename = os.path.split(file_path)[1]

        if from_layer:
            self.log(f"\nLayer prediction saved as : {filename}")
        else:
            self.log(f"\nFile n°{i+1} saved as : {filename}")

    def aniso_transform(self, image):
        zoom = self.transforms["zoom"][1]
        anisotropic_transform = Zoom(
            zoom=zoom,
            keep_size=False,
            padding_mode="empty",
        )
        return anisotropic_transform(image[0])

    def instance_seg(self, to_instance, image_id=0, original_filename="layer"):

        if image_id is not None:
            self.log(f"\nRunning instance segmentation for image n°{image_id}")

        threshold = self.instance_params["threshold"]
        size_small = self.instance_params["size_small"]
        method_name = self.instance_params["method"]

        if method_name == "Watershed":

            def method(image):
                return binary_watershed(image, threshold, size_small)

        elif method_name == "Connected components":

            def method(image):
                return binary_connected(image, threshold, size_small)

        else:
            raise NotImplementedError(
                "Selected instance segmentation method is not defined"
            )

        instance_labels = method(to_instance)

        instance_filepath = (
            self.results_path
            + "/"
            + f"Instance_seg_labels_{image_id}_"
            + original_filename
            + "_"
            + self.model_dict["name"]
            + f"_{utils.get_date_time()}_"
            + self.filetype
        )

        imwrite(instance_filepath, instance_labels)
        self.log(
            f"Instance segmentation results for image n°{image_id} have been saved as:"
        )
        self.log(os.path.split(instance_filepath)[1])
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
        instance_labels, data_dict = self.get_instance_result(out, i=i)

        original = np.array(inf_data["image"]).astype(np.float32)

        self.log(f"Inference completed on layer")

        return self.create_result_dict(
            out,
            instance_labels,
            from_layer=False,
            original=original,
            data_dict=data_dict,
            i=i,
        )

    def stats_csv(self, instance_labels):
        if self.stats_to_csv:

            # try:

            data_dict = volume_stats(
                instance_labels
            )  # TODO test with area mesh function
            return data_dict

        # except ValueError as e:
        #     self.log(f"Error occurred during stats computing : {e}")
        #     return None
        else:
            return None

    def inference_on_layer(self, image, model, post_process_transforms):
        self.log("-" * 10)
        self.log(f"Inference started on layer...")

        image = image.type(torch.FloatTensor)

        out = self.model_output(
            image,
            model,
            post_process_transforms,
            aniso_transform=self.aniso_transform,
        )

        self.save_image(out, from_layer=True)

        instance_labels, data_dict = self.get_instance_result(
            out, from_layer=True
        )

        return self.create_result_dict(
            out, instance_labels, from_layer=True, data_dict=data_dict
        )

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
        print(f"OS is {sys}")
        if sys == "Darwin":
            torch.set_num_threads(1)  # required for threading on macOS ?
            self.log("Number of threads has been set to 1 for macOS")

        try:
            dims = self.model_dict["model_input_size"]
            self.log(f"MODEL DIMS : {dims}")
            self.log(self.model_dict["name"])

            if self.model_dict["name"] == "SegResNet":
                model = self.model_dict["class"].get_net(
                    input_image_size=[
                        dims,
                        dims,
                        dims,
                    ],  # TODO FIX ! find a better way & remove model-specific code
                )
            elif self.model_dict["name"] == "SwinUNetR":
                model = self.model_dict["class"].get_net(
                    img_size=[dims, dims, dims],
                    use_checkpoint=False,
                )
            else:
                model = self.model_dict["class"].get_net()
            model = model.to(self.device)

            self.log_parameters()

            model.to(self.device)

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

            if not self.transforms["thresh"][0]:
                post_process_transforms = EnsureType()
            else:
                t = self.transforms["thresh"][1]
                post_process_transforms = Compose(
                    AsDiscrete(threshold=t), EnsureType()
                )

            self.log("\nLoading weights...")

            if self.weights_dict["custom"]:
                weights = self.weights_dict["path"]
            else:
                self.downloader.download_weights(
                    self.model_dict["name"],
                    self.model_dict["class"].get_weights_file(),
                )
                weights = os.path.join(
                    WEIGHTS_DIR, self.model_dict["class"].get_weights_file()
                )

            model.load_state_dict(
                torch.load(
                    weights,
                    map_location=self.device,
                )
            )
            self.log("Done")

            is_folder = self.images_filepaths is not None
            is_layer = self.layer is not None

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
                # print(image.shape)
                ##################
                ##################
            elif is_layer:
                input_image = self.load_layer()
            else:
                raise ValueError("No data has been provided. Aborting.")

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

        except Exception as e:
            self.log(f"Error : {e}")
            self.quit()
        finally:
            self.quit()


class TrainingWorker(GeneratorWorker):
    """A custom worker to run training jobs in.
    Inherits from :py:class:`napari.qt.threading.GeneratorWorker`"""

    def __init__(
        self,
        device,
        model_dict,
        weights_path,
        data_dicts,
        validation_percent,
        max_epochs,
        loss_function,
        learning_rate,
        val_interval,
        batch_size,
        results_path,
        sampling,
        num_samples,
        sample_size,
        do_augmentation,
        deterministic,
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

        self._weight_error = False
        #############################################
        self.device = device
        self.model_dict = model_dict
        self.weights_path = weights_path
        self.data_dicts = data_dicts
        self.validation_percent = validation_percent
        self.max_epochs = max_epochs
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.val_interval = val_interval
        self.batch_size = batch_size
        self.results_path = results_path

        self.num_samples = num_samples
        self.sampling = sampling
        self.sample_size = sample_size

        self.do_augment = do_augmentation
        self.seed_dict = deterministic

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

    def log_parameters(self):

        self.log("-" * 20)
        self.log("Parameters summary :\n")

        self.log(
            f"Percentage of dataset used for validation : {self.validation_percent * 100}%"
        )
        self.log("-" * 10)
        self.log("Training files :\n")
        [
            self.log(f"{os.path.basename(str(train_file)[:-2])}\n")
            for train_file in self.train_files
        ]
        self.log("-" * 10)
        self.log("Validation files :\n")
        [
            self.log(f"{os.path.basename(str(val_file)[:-2])}\n")
            for val_file in self.val_files
        ]
        self.log("-" * 10)
        if self.seed_dict["use deterministic"]:
            self.log(f"Deterministic training is enabled")
            self.log(f"Seed is {self.seed_dict['seed']}")

        self.log(f"Training for {self.max_epochs} epochs")
        self.log(f"Loss function is : {str(self.loss_function)}")
        self.log(f"Validation is performed every {self.val_interval} epochs")
        self.log(f"Batch size is {self.batch_size}")
        self.log(f"Learning rate is {self.learning_rate}")

        if self.sampling:
            self.log(
                f"Extracting {self.num_samples} patches of size {self.sample_size}"
            )
        else:
            self.log("Using whole images as dataset")

        if self.do_augment:
            self.log("Data augmentation is enabled")

        if self.weights_path is not None:
            self.log(f"Using weights from : {self.weights_path}")
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
        try:
            if self.seed_dict["use deterministic"]:
                set_determinism(
                    seed=self.seed_dict["seed"]
                )  # use_deterministic_algorithms = True causes cuda error

            sys = platform.system()
            print(sys)
            if sys == "Darwin":  # required for macOS ?
                torch.set_num_threads(1)
                self.log("Number of threads has been set to 1 for macOS")

            model_name = self.model_dict["name"]
            model_class = self.model_dict["class"]

            if not self.sampling:
                data_check = LoadImaged(keys=["image"])(self.data_dicts[0])
                check = data_check["image"].shape

            if model_name == "SegResNet":
                if self.sampling:
                    size = self.sample_size
                else:
                    size = check
                print(f"Size of image : {size}")
                model = model_class.get_net(
                    input_image_size=utils.get_padding_dim(size),
                    out_channels=1,
                    dropout_prob=0.3,
                )
            elif model_name == "SwinUNetR":
                if self.sampling:
                    size = self.sample_size
                else:
                    size = check
                print(f"Size of image : {size}")
                model = model_class.get_net(
                    img_size=utils.get_padding_dim(size),
                    use_checkpoint=True,
                )
            else:
                model = model_class.get_net()  # get an instance of the model
            model = model.to(self.device)

            epoch_loss_values = []
            val_metric_values = []

            self.train_files, self.val_files = (
                self.data_dicts[
                    0 : int(len(self.data_dicts) * self.validation_percent)
                ],
                self.data_dicts[
                    int(len(self.data_dicts) * self.validation_percent) :
                ],
            )

            if self.train_files == [] or self.val_files == []:
                self.log("ERROR : datasets are empty")

            if self.sampling:
                sample_loader = Compose(
                    [
                        LoadImaged(keys=["image", "label"]),
                        EnsureChannelFirstd(keys=["image", "label"]),
                        RandSpatialCropSamplesd(
                            keys=["image", "label"],
                            roi_size=(
                                self.sample_size
                            ),  # multiply by axis_stretch_factor if anisotropy
                            # max_roi_size=(120, 120, 120),
                            random_size=False,
                            num_samples=self.num_samples,
                        ),
                        Orientationd(keys=["image", "label"], axcodes="PLI"),
                        SpatialPadd(
                            keys=["image", "label"],
                            spatial_size=(
                                utils.get_padding_dim(self.sample_size)
                            ),
                        ),
                        EnsureTyped(keys=["image", "label"]),
                    ]
                )

            if self.do_augment:
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
            if self.sampling:
                print("train_ds")
                train_ds = PatchDataset(
                    data=self.train_files,
                    transform=train_transforms,
                    patch_func=sample_loader,
                    samples_per_image=self.num_samples,
                )
                print("val_ds")
                val_ds = PatchDataset(
                    data=self.val_files,
                    transform=val_transforms,
                    patch_func=sample_loader,
                    samples_per_image=self.num_samples,
                )

            else:
                load_single_images = Compose(
                    [
                        LoadImaged(keys=["image", "label"]),
                        EnsureChannelFirstd(keys=["image", "label"]),
                        Orientationd(keys=["image", "label"], axcodes="PLI"),
                        SpatialPadd(
                            keys=["image", "label"],
                            spatial_size=(utils.get_padding_dim(check)),
                        ),
                        EnsureTyped(keys=["image", "label"]),
                    ]
                )
                print("Cache dataset : train")
                train_ds = CacheDataset(
                    data=self.train_files,
                    transform=Compose(load_single_images, train_transforms),
                )
                print("Cache dataset : val")
                val_ds = CacheDataset(
                    data=self.val_files, transform=load_single_images
                )
            print("Dataloader")
            train_loader = DataLoader(
                train_ds,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=2,
                collate_fn=pad_list_data_collate,
            )

            val_loader = DataLoader(
                val_ds, batch_size=self.batch_size, num_workers=2
            )
            print("\nDone")

            print("Optimizer")
            optimizer = torch.optim.Adam(
                model.parameters(), self.learning_rate
            )
            dice_metric = DiceMetric(include_background=True, reduction="mean")

            best_metric = -1
            best_metric_epoch = -1

            # time = utils.get_date_time()
            print("Weights")
            if self.weights_path is not None:
                if self.weights_path == "use_pretrained":
                    weights_file = model_class.get_weights_file()
                    self.downloader.download_weights(model_name, weights_file)
                    weights = os.path.join(WEIGHTS_DIR, weights_file)
                    self.weights_path = weights
                else:
                    weights = os.path.join(self.weights_path)

                try:
                    model.load_state_dict(
                        torch.load(
                            weights,
                            map_location=self.device,
                        )
                    )
                except RuntimeError as e:
                    print(f"Error : {e}")
                    warn = (
                        "WARNING:\nIt'd seem that the weights were incompatible with the model,\n"
                        "the model will be trained from random weights"
                    )
                    self.log(warn)
                    self.warn(warn)
                    self._weight_error = True

            if self.device.type == "cuda":
                self.log("\nUsing GPU :")
                self.log(torch.cuda.get_device_name(0))
            else:
                self.log("Using CPU")

            self.log_parameters()

            for epoch in range(self.max_epochs):
                # self.log("\n")
                self.log("-" * 10)
                self.log(f"Epoch {epoch + 1}/{self.max_epochs}")
                if self.device.type == "cuda":
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
                        batch_data["image"].to(self.device),
                        batch_data["label"].to(self.device),
                    )
                    optimizer.zero_grad()
                    outputs = model_class.get_output(model, inputs)
                    # print(f"OUT : {outputs.shape}")
                    loss = self.loss_function(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.detach().item()
                    self.log(
                        f"* {step}/{len(train_ds) // train_loader.batch_size}, "
                        f"Train loss: {loss.detach().item():.4f}"
                    )
                    yield {"plot": False, "weights": model.state_dict()}

                epoch_loss /= step
                epoch_loss_values.append(epoch_loss)
                self.log(f"Epoch: {epoch + 1}, Average loss: {epoch_loss:.4f}")

                if (epoch + 1) % self.val_interval == 0:
                    model.eval()
                    with torch.no_grad():
                        for val_data in val_loader:
                            val_inputs, val_labels = (
                                val_data["image"].to(self.device),
                                val_data["label"].to(self.device),
                            )

                            val_outputs = model_class.get_validation(
                                model, val_inputs
                            )

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

                            # print(len(val_outputs))
                            # print(len(val_labels))

                            dice_metric(y_pred=val_outputs, y=val_labels)

                        metric = dice_metric.aggregate().detach().item()
                        dice_metric.reset()

                        val_metric_values.append(metric)

                        train_report = {
                            "plot": True,
                            "epoch": epoch,
                            "losses": epoch_loss_values,
                            "val_metrics": val_metric_values,
                            "weights": model.state_dict(),
                        }
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
                                os.path.join(
                                    self.results_path, weights_filename
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
            self.log(f"Error : {e}")
            self.quit()
        finally:
            self.quit()
