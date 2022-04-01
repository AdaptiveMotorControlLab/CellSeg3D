import glob
import os
import torch

import napari
from napari_cellseg_annotator import utils
from qtpy.QtWidgets import (
    QWidget,
    QPushButton,
    QSizePolicy,
    QLabel,
    QComboBox,
)


class ModelFramework(QWidget):
    def __init__(self, viewer: "napari.viewer.Viewer"):

        super().__init__()

        self._viewer = viewer

        self.model_type = None

        self.images_filepaths = ""
        self.labels_filepaths = ""
        self.results_path = ""
        self.model_path = ""

        self.device = "cpu"

        self._default_path = [self.images_filepaths, self.labels_filepaths]
        self._default_model_path = [self.model_path]
        self._default_res_path = [self.results_path]

        #######################################################
        # interface
        self.btn_image_files = QPushButton("Open", self)
        self.btn_image_files.setSizePolicy(
            QSizePolicy.Fixed, QSizePolicy.Fixed
        )
        self.lbl_image_files = QLabel("Images directory", self)
        self.btn_image_files.clicked.connect(self.load_image_dataset)

        self.btn_label_files = QPushButton("Open", self)
        self.btn_label_files.setSizePolicy(
            QSizePolicy.Fixed, QSizePolicy.Fixed
        )
        self.lbl_label_files = QLabel("Images directory", self)
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
        self.lbl_result_path = QLabel("Results directory", self)
        self.btn_result_path.clicked.connect(self.load_results_path)

        self.btn_model_path = QPushButton("Open", self)
        self.btn_model_path.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.lbl_model_path = QLabel("Model directory", self)
        self.btn_model_path.clicked.connect(self.load_label_dataset)

        self.btn_close = QPushButton("Close", self)
        self.btn_close.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn_close.clicked.connect(self.close)
        #######################################################

    def update_default(self):
        self._default_path = [self.images_filepaths, self.labels_filepaths]
        self._default_model_path = [self.model_path]
        self._default_res_path = [self.results_path]

    def load_dataset_paths(self):
        filetype = self.filetype_choice.currentText()
        directory = utils.open_file_dialog(self, self._default_path, True)
        # print(directory)
        file_paths = sorted(glob.glob(os.path.join(directory, "*" + filetype)))
        # print(file_paths)
        return file_paths

    def create_train_dataset_dict(self):

        data_dicts = [
            {"image": image_name, "label": label_name}
            for image_name, label_name in zip(
                self.images_filepaths, self.labels_filepaths
            )
        ]

        return data_dicts

    def load_image_dataset(self):
        filenames = self.load_dataset_paths()
        if filenames != "" and filenames != []:
            self.images_filepaths = filenames
            # print(filenames)
            path = os.path.dirname(filenames[0])
            self.lbl_image_files.setText(path)
            self.update_default()

    def load_label_dataset(self):
        filenames = self.load_dataset_paths()
        if filenames != "":
            self.labels_filepaths = filenames
            path = os.path.dirname(filenames[0])
            self.lbl_label_files.setText(path)
            self.update_default()

    def load_results_path(self):
        dir = utils.open_file_dialog(self, self._default_res_path, True)
        if dir != "" and type(dir) is str:
            self.results_path = dir
            self.lbl_result_path.setText(self.results_path)
            self.update_default()

    def load_model_path(self):
        dir = utils.open_file_dialog(self, self._default_model_path)
        if dir != "" and type(dir) is str:
            self.model_path = dir
            self.lbl_model_path.setText(self.results_path)
            self.update_default()

    def get_device(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"Using {self.device} device")
        print("Using torch :")
        print(torch.__version__)

    def get_padding_dim(self, image_shape):
        padding = []
        for p in range(3):
            n = 0
            pad = -1
            while pad < image_shape[p]:
                pad = 2**n
                n += 1
                if pad > 4095:
                    return
            padding.append(pad)
        return padding

    def transform(self):
        return

    def train(self):
        return

    def build(self):
        raise NotImplementedError("Should be defined in children classes")

    def close(self):
        self._viewer.window.remove_dock_widget(self)
