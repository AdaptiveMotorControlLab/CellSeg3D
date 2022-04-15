import glob
import os
import warnings

import napari
import torch
from qtpy.QtWidgets import (
    QTabWidget,
    QPushButton,
    QSizePolicy,
    QLabel,
    QComboBox,
    QLineEdit,
)

from napari_cellseg_annotator import utils
from napari_cellseg_annotator.models import model_SegResNet as SegResNet
from napari_cellseg_annotator.models import model_VNet as VNet
from napari_cellseg_annotator.models import TRAILMAP_test as TMAP

warnings.formatwarning = utils.format_Warning


class ModelFramework(QTabWidget):
    """A framework with buttons to use for loading images, labels, models, etc. for both inference and training"""

    def __init__(self, viewer: "napari.viewer.Viewer"):
        """Creates a plugin framework with the following elements :

        * A button to choose an image folder containing the images of a dataset (e.g. dataset/images)

        * A button to choose a label folder containing the labels of a dataset (e.g. dataset/labels)

        * A button to choose a results folder to save results in (e.g. dataset/inference_results)

        * A file extension choice to choose which file types to load in the data folders

        Args:
            viewer (napari.viewer.Viewer): viewer to load the widget in
        """
        super().__init__()

        self._viewer = viewer
        """napari.viewer.Viewer: Viewer to display the widget in in"""

        self.images_filepaths = [""]
        """array(str): paths to images for training or inference"""
        self.labels_filepaths = [""]
        """array(str): paths to labels for training"""
        self.results_path = ""
        """str: path to output folder,to save results in"""
        self.model_path = ""
        """str: path to custom model defined by user"""

        self.device = "cpu"
        """Device to train on, chosen automatically by :py:func:`get_device`"""

        self.models_dict = {
            "VNet": VNet,
            "SegResNet": SegResNet,
            "TRAILMAP test": TMAP,
        }
        """dict: dictionary of available models, with string for widget display as key

        Currently implemented : SegResNet, VNet, TRAILMAP_test"""

        self._default_path = [
            self.images_filepaths,
            self.labels_filepaths,
            self.model_path,
            self.results_path,
        ]

        #######################################################
        # interface
        self.btn_image_files = QPushButton("Open", self)
        self.btn_image_files.setSizePolicy(
            QSizePolicy.Fixed, QSizePolicy.Fixed
        )
        self.lbl_image_files = QLineEdit("Images directory", self)
        self.lbl_image_files.setReadOnly(True)
        self.btn_image_files.clicked.connect(self.load_image_dataset)

        self.btn_label_files = QPushButton("Open", self)
        self.btn_label_files.setSizePolicy(
            QSizePolicy.Fixed, QSizePolicy.Fixed
        )
        self.lbl_label_files = QLineEdit("Labels directory", self)
        self.lbl_label_files.setReadOnly(True)
        self.btn_label_files.clicked.connect(self.load_label_dataset)

        self.filetype_choice = QComboBox()
        self.filetype_choice.addItems([".tif", ".tiff"])
        self.filetype_choice.setSizePolicy(
            QSizePolicy.Fixed, QSizePolicy.Fixed
        )
        self.lbl_filetype = QLabel("Filetype :", self)

        self.btn_result_path = QPushButton("Open", self)
        self.btn_result_path.setSizePolicy(
            QSizePolicy.Fixed, QSizePolicy.Fixed
        )
        self.lbl_result_path = QLineEdit("Results directory", self)
        self.lbl_result_path.setReadOnly(True)
        self.btn_result_path.clicked.connect(self.load_results_path)

        self.btn_model_path = QPushButton("Open", self)
        self.btn_model_path.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.lbl_model_path = QLineEdit("Model directory", self)
        self.lbl_model_path.setReadOnly(True)
        self.btn_model_path.clicked.connect(self.load_label_dataset)

        self.model_choice = QComboBox()
        self.model_choice.addItems(sorted(self.models_dict.keys()))
        self.lbl_model_choice = QLabel("Model name", self)

        self.btn_prev = QPushButton()
        self.btn_prev.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn_prev.setText("Previous")
        self.btn_prev.clicked.connect(
            lambda: self.setCurrentIndex(self.currentIndex() - 1)
        )

        self.btn_next = QPushButton()
        self.btn_next.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn_next.setText("Next")
        self.btn_next.clicked.connect(
            lambda: self.setCurrentIndex(self.currentIndex() + 1)
        )

        self.btn_close = QPushButton("Close", self)
        self.btn_close.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn_close.clicked.connect(self.close)
        #######################################################

    def update_default(self):
        """Update default path for smoother file dialogs"""
        self._default_path = [
            path
            for path in [
                os.path.dirname(self.images_filepaths[0]),
                os.path.dirname(self.labels_filepaths[0]),
                self.model_path,
                self.results_path,
            ]
            if (path != [""] and path != "")
        ]

    def load_dataset_paths(self):
        """Loads all image paths (as str) in a given folder for which the extension matches the set filetype

        Returns:
           array(str): all loaded file paths
        """
        filetype = self.filetype_choice.currentText()
        directory = utils.open_file_dialog(self, self._default_path, True)
        # print(directory)
        file_paths = sorted(glob.glob(os.path.join(directory, "*" + filetype)))
        # print(file_paths)
        return file_paths

    def create_train_dataset_dict(self):
        """Creates data dictionary for MONAI transforms and training.

        **Keys:**

        * "image": image

        * "label" : corresponding label
        """

        print("Images :\n")
        for file in self.images_filepaths:
            print(os.path.basename(file).split(".")[0])
        print("*" * 10)
        print("\nLabels :\n")
        for file in self.labels_filepaths:
            print(os.path.basename(file).split(".")[0])

        data_dicts = [
            {"image": image_name, "label": label_name}
            for image_name, label_name in zip(
                self.images_filepaths, self.labels_filepaths
            )
        ]

        return data_dicts

    def get_model(self, key):
        """Getter for module (class and functions) associated to currently selected model"""
        return self.models_dict[key]

    def get_loss(self, key):
        """Getter for loss function selected by user"""
        return self.loss_dict[key]

    def load_image_dataset(self):
        """Show file dialog to set :py:attr:`images_filepaths`"""
        filenames = self.load_dataset_paths()
        # print(filenames)
        if filenames != "" and filenames != [""]:
            self.images_filepaths = filenames
            # print(filenames)
            path = os.path.dirname(filenames[0])
            self.lbl_image_files.setText(path)
            # print(path)
            self._default_path[0] = path

    def load_label_dataset(self):
        """Show file dialog to set :py:attr:`labels_filepaths`"""
        filenames = self.load_dataset_paths()
        if filenames != "" and filenames != [""]:
            self.labels_filepaths = filenames
            path = os.path.dirname(filenames[0])
            self.lbl_label_files.setText(path)
            self.update_default()

    def load_results_path(self):
        """Show file dialog to set :py:attr:`results_path`"""
        dir = utils.open_file_dialog(self, self._default_path, True)
        if dir != "" and type(dir) is str and os.path.isdir(dir):
            self.results_path = dir
            self.lbl_result_path.setText(self.results_path)
            self.update_default()

    def load_model_path(self):
        """Show file dialog to set :py:attr:`model_path`"""
        dir = utils.open_file_dialog(self, self._default_path)
        if dir != "" and type(dir) is str and os.path.isdir(dir):
            self.model_path = dir
            self.lbl_model_path.setText(self.results_path)
            self.update_default()

    def get_device(self, show = True):
        """Automatically discovers any cuda device and uses it for tensor operations.
        If none is available (CUDA not installed), uses cpu instead."""
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        if show:
            print(f"Using {self.device} device")
            print("Using torch :")
            print(torch.__version__)
        return self.device

    def empty_cuda_cache(self):
        print("Empyting cache...")
        torch.cuda.empty_cache()
        print("Cache emptied")

    def get_padding_dim(self, image_shape):
        """
        Finds the nearest and superior power of two for each image dimension to pad it for CNN processing,
        for either 2D or 3D images. E.g. an image size of 30x40x100 will result in a padding of 32x64x128.
        Shows a warning if the padding dimensions are very large.


        Args:
            image_shape (torch.size): an array of the dimensions of the image in D/H/W if 3D or H/W if 2D

        Returns:
            array(int): padding value for each dim
        """
        padding = []

        dims = len(image_shape)
        print(f"Dimension of data for padding : {dims}D")
        if dims != 2 and dims != 3:
            raise ValueError(
                "Please check the dimensions of the input, only 2 or 3-dimensional data is supported currently"
            )

        for p in range(dims):
            n = 0
            pad = -1
            while pad < image_shape[p]:
                pad = 2**n
                n += 1
                if pad >= 1024:
                    warnings.warn(
                        "Warning : a very large dimension for automatic padding has been computed.\n"
                        "Ensure your images are of an appropriate size and/or that you have enough memory."
                        f"The padding value is currently {pad}."
                    )

            padding.append(pad)
        return padding

    def build(self):
        raise NotImplementedError("Should be defined in children classes")

    def close(self):
        """Removes widget from viewer"""
        self._viewer.window.remove_dock_widget(self)
