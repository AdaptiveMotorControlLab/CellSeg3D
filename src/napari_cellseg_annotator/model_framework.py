import glob
import os
import warnings

import napari
import torch
from qtpy.QtWidgets import QComboBox
from qtpy.QtWidgets import QLabel
from qtpy.QtWidgets import QLineEdit
from qtpy.QtWidgets import QProgressBar
from qtpy.QtWidgets import QPushButton
from qtpy.QtWidgets import QSizePolicy
from qtpy.QtWidgets import QTabWidget
from qtpy.QtWidgets import QTextEdit
from qtpy.QtWidgets import QVBoxLayout
from qtpy.QtWidgets import QWidget

from napari_cellseg_annotator import utils
from napari_cellseg_annotator.models import TRAILMAP_test as TMAP
from napari_cellseg_annotator.models import model_SegResNet as SegResNet
from napari_cellseg_annotator.models import model_VNet as VNet

warnings.formatwarning = utils.format_Warning


class ModelFramework(QTabWidget):
    """A framework with buttons to use for loading images, labels, models, etc. for both inference and training"""

    def __init__(self, viewer: "napari.viewer.Viewer"):
        """Creates a plugin framework with the following elements :

        * A button to choose an image folder containing the images of a dataset (e.g. dataset/images)

        * A button to choose a label folder containing the labels of a dataset (e.g. dataset/labels)

        * A button to choose a results folder to save results in (e.g. dataset/inference_results)

        * A file extension choice to choose which file types to load in the data folders

        * A docked module with a log, a progress bar and a save button (see :py:func:`~display_status_report`)

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

        self.docked_widgets = []

        self.worker = None

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
        self.lbl_filetype = QLabel("File type", self)

        self.btn_result_path = QPushButton("Open", self)
        self.btn_result_path.setSizePolicy(
            QSizePolicy.Fixed, QSizePolicy.Fixed
        )
        self.lbl_result_path = QLineEdit("Results directory", self)
        self.lbl_result_path.setReadOnly(True)
        self.btn_result_path.clicked.connect(self.load_results_path)

        # TODO : implement custom model
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

        ###################################################
        # status report docked widget
        self.container_report = QWidget()
        self.container_report.setSizePolicy(
            QSizePolicy.Fixed, QSizePolicy.Minimum
        )
        self.container_docked = False  # check if already docked

        self.progress = QProgressBar(self.container_report)
        self.progress.setVisible(False)
        """Widget for the progress bar"""

        self.log = QTextEdit(self.container_report)
        self.log.setVisible(False)
        """Read-only display for process-related info. Use only for info destined to user."""

        self.btn_save_log = QPushButton(
            "Save log with results", self.container_report
        )
        self.btn_save_log.clicked.connect(self.save_log)
        self.btn_save_log.setVisible(False)
        #####################################################

    def save_log(self):
        """Saves the worker's log to disk at self.results_path when called"""
        log = self.log.toPlainText()

        if len(log) != 0:
            with open(
                self.results_path + f"/Log_report_{utils.get_date_time()}.txt",
                "x",
            ) as f:
                f.write(log)
                f.close()
        else:
            warnings.warn(
                "No job has been completed yet, please start one or re-open the log window."
            )

    def print_and_log(self, text):
        """Utility used to both print to terminal and log action to self.log. Use only for important user info.

        Args:
            text (str): Text to be printed and logged

        """
        print(text)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())
        self.log.insertPlainText(f"\n{text}")

    @staticmethod
    def worker_print_and_log(widget, text):
        """Utility used to both print to terminal and log action to self.log, modified to be usable in a static worker.
        Use only for important user info.

             Args:
                 widget (QWidget): widget with a self.log (QTextEdit) attribute

                 text (str): Text to be printed and logged

        """
        # TODO : fix warning for cursor instance
        print(text)
        # widget.log.moveCursor(QTextCursor.End)
        widget.log.verticalScrollBar().setValue(widget.log.verticalScrollBar().maximum())
        widget.log.insertPlainText(f"\n{text}")

    def display_status_report(self):
        """Adds a text log, a progress bar and a "save log" button on the left side of the viewer (usually when starting a worker)"""

        if self.container_report is None or self.log is None:
            warnings.warn(
                "Status report widget has been closed. Trying to re-instantiate..."
            )
            self.container_report = QWidget()
            self.container_report.setSizePolicy(
                QSizePolicy.Fixed, QSizePolicy.Minimum
            )
            self.progress = QProgressBar(self.container_report)
            self.log = QTextEdit(self.container_report)
            self.btn_save_log = QPushButton(
                "Save log with results", self.container_report
            )
            self.btn_save_log.clicked.connect(self.save_log)

            self.container_docked = False  # check if already docked

        if self.container_docked:
            self.log.clear()
        elif not self.container_docked:
            temp_layout = QVBoxLayout()
            temp_layout.setContentsMargins(10, 5, 5, 5)

            temp_layout.addWidget(  # DO NOT USE alignment here, it will break auto-resizing
                self.progress  # , alignment=utils.CENTER_AL
            )
            temp_layout.addWidget(self.log)  # , alignment=utils.CENTER_AL
            temp_layout.addWidget(
                self.btn_save_log  # , alignment=utils.CENTER_AL
            )
            self.container_report.setLayout(temp_layout)

            report_dock = self._viewer.window.add_dock_widget(
                self.container_report,
                name="Status report",
                area="left",
                allowed_areas=["left"],
            )
            self.docked_widgets.append(report_dock)
            self.container_docked = True

        self.log.setVisible(True)
        self.progress.setVisible(True)
        self.btn_save_log.setVisible(True)
        self.progress.setValue(0)

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

        Returns: a dict with the following :
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
        """Show file dialog to set :py:attr:`~images_filepaths`"""
        filenames = self.load_dataset_paths()
        # print(filenames)
        if filenames != "" and filenames != [""] and filenames != []:
            self.images_filepaths = filenames
            # print(filenames)
            path = os.path.dirname(filenames[0])
            self.lbl_image_files.setText(path)
            # print(path)
            self._default_path[0] = path

    def load_label_dataset(self):
        """Show file dialog to set :py:attr:`~labels_filepaths`"""
        filenames = self.load_dataset_paths()
        if filenames != "" and filenames != [""]:
            self.labels_filepaths = filenames
            path = os.path.dirname(filenames[0])
            self.lbl_label_files.setText(path)
            self.update_default()

    def load_results_path(self):
        """Show file dialog to set :py:attr:`~results_path`"""
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

    @staticmethod
    def get_device(show=True):
        """Automatically discovers any cuda device and uses it for tensor operations.
        If none is available (CUDA not installed), uses cpu instead."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if show:
            print(f"Using {device} device")
            print("Using torch :")
            print(torch.__version__)
        return device

    def empty_cuda_cache(self):
        """Empties the cuda cache if the device is a cuda device"""
        if self.get_device(show=False).type == "cuda":
            print("Empyting cache...")
            torch.cuda.empty_cache()
            print("Cache emptied")

    def build(self):
        raise NotImplementedError("Should be defined in children classes")

    def close(self):
        """Close the widget and the docked widgets, if any"""
        if len(self.docked_widgets) != 0:
            [
                self._viewer.window.remove_dock_widget(w)
                for w in self.docked_widgets
            ]
        self._viewer.window.remove_dock_widget(self)
