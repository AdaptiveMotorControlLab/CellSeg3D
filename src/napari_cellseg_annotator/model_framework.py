import glob
import os

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
        self.filetype = ""
        self.results_path = ""
        self.model_path = ""

        self._default_path = [self.images_filepaths, self.labels_filepaths]

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

    def load_results_path(self):
        self.results_path = utils.open_file_dialog(self, self._default_path)

    def load_model_path(self):
        self.model_path = utils.open_file_dialog(self, self._default_path)

    def load_dataset_paths(self):
        filetype = self.filetype_choice.currentText()
        directory = utils.open_file_dialog(self, self._default_path, True)
        file_paths = sorted(
            glob.glob(os.path.join(directory, ("*" + filetype)))
        )

        return file_paths

    def create_train_dataset_dict(self, images_paths, labels_paths):

        data_dicts = [
            {"image": image_name, "label": label_name}
            for image_name, label_name in zip(images_paths, labels_paths)
        ]

        return data_dicts

    def load_image_dataset(self):
        self.images_filepaths = self.load_dataset_paths()

    def load_label_dataset(self):
        self.labels_filepaths = self.load_dataset_paths()

    def transform(self):
        return

    def train(self):
        return

    def build(self):
        raise NotImplementedError("Should be defined in children classes")

    def close(self):
        self._viewer.window.remove_dock_widget(self)
