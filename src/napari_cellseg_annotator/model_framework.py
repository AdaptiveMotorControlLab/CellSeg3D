import glob
import os
import warnings

import napari
import torch
# Qt
from qtpy.QtWidgets import QLineEdit
from qtpy.QtWidgets import QProgressBar
from qtpy.QtWidgets import QSizePolicy
from qtpy.QtWidgets import QTabWidget

# local
from napari_cellseg_annotator import interface as ui
from napari_cellseg_annotator import utils
from napari_cellseg_annotator.log_utility import Log
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
        """List of docked widgets (returned by :py:func:`viewer.window.add_dock_widget()),
        can be used to remove docked widgets`"""

        self.worker = None
        """Worker from model_workers.py, either inference or training"""

        #######################################################
        # interface
        self.btn_image_files = ui.make_button(
            "Open", self.load_image_dataset, self
        )
        self.lbl_image_files = QLineEdit("Images directory", self)
        self.lbl_image_files.setReadOnly(True)

        self.btn_label_files = ui.make_button(
            "Open", self.load_label_dataset, self
        )
        self.lbl_label_files = QLineEdit("Labels directory", self)
        self.lbl_label_files.setReadOnly(True)

        self.filetype_choice, self.lbl_filetype = ui.make_combobox(
            [".tif", ".tiff"], label="File format"
        )

        self.btn_result_path = ui.make_button(
            "Open", self.load_results_path, self
        )
        self.lbl_result_path = QLineEdit("Results directory", self)
        self.lbl_result_path.setReadOnly(True)

        # TODO : implement custom model
        self.btn_model_path = ui.make_button(
            "Open", self.load_label_dataset, self
        )
        self.lbl_model_path = QLineEdit("Model directory", self)
        self.lbl_model_path.setReadOnly(True)

        self.model_choice, self.lbl_model_choice = ui.make_combobox(
            sorted(self.models_dict.keys()), label="Model name"
        )
        ###################################################
        # status report docked widget
        (
            self.container_report,
            self.container_report_layout,
        ) = ui.make_container_widget(10, 5, 5, 5)
        self.container_report.setSizePolicy(
            QSizePolicy.Fixed, QSizePolicy.Minimum
        )
        self.container_docked = False  # check if already docked

        self.progress = QProgressBar(self.container_report)
        self.progress.setVisible(False)
        """Widget for the progress bar"""

        self.log = Log(self.container_report)
        self.log.setVisible(False)
        """Read-only display for process-related info. Use only for info destined to user."""

        self.btn_save_log = ui.make_button(
            "Save log in results folder",
            self.save_log,
            self.container_report,
            fixed=False,
        )
        self.btn_save_log.setVisible(False)
        #####################################################

    def make_close_button(self):
        btn = ui.make_button("Close", self.close)
        return btn

    def make_prev_button(self):
        btn = ui.make_button(
            "Previous", lambda: self.setCurrentIndex(self.currentIndex() - 1)
        )
        return btn

    def make_next_button(self):
        btn = ui.make_button(
            "Next", lambda: self.setCurrentIndex(self.currentIndex() + 1)
        )
        return btn

    def send_log(self, text):
        self.log.print_and_log(text)

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

    def display_status_report(self):
        """Adds a text log, a progress bar and a "save log" button on the left side of the viewer (usually when starting a worker)"""

        # if self.container_report is None or self.log is None:
        #     warnings.warn(
        #         "Status report widget has been closed. Trying to re-instantiate..."
        #     )
        #     self.container_report = QWidget()
        #     self.container_report.setSizePolicy(
        #         QSizePolicy.Fixed, QSizePolicy.Minimum
        #     )
        #     self.progress = QProgressBar(self.container_report)
        #     self.log = QTextEdit(self.container_report)
        #     self.btn_save_log = ui.make_button(
        #         "Save log in results folder", parent=self.container_report
        #     )
        #     self.btn_save_log.clicked.connect(self.save_log)
        #
        #     self.container_docked = False  # check if already docked

        if self.container_docked:
            self.log.clear()
        elif not self.container_docked:

            self.container_report_layout.addWidget(  # DO NOT USE alignment here, it will break auto-resizing
                self.progress  # , alignment=ui.CENTER_AL
            )
            self.container_report_layout.addWidget(
                self.log
            )  # , alignment=ui.CENTER_AL
            self.container_report_layout.addWidget(
                self.btn_save_log  # , alignment=ui.CENTER_AL
            )
            self.container_report.setLayout(self.container_report_layout)

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
        directory = ui.open_file_dialog(self, self._default_path, True)
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
        dir = ui.open_file_dialog(self, self._default_path, True)
        if dir != "" and type(dir) is str and os.path.isdir(dir):
            self.results_path = dir
            self.lbl_result_path.setText(self.results_path)
            self.update_default()

    def load_model_path(self):
        """Show file dialog to set :py:attr:`model_path`"""
        dir = ui.open_file_dialog(self, self._default_path)
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

    def remove_docked_widgets(self):
        """Removes docked widgets and resets checks for status report"""
        if len(self.docked_widgets) != 0:
            [
                self._viewer.window.remove_dock_widget(w)
                for w in self.docked_widgets
            ]
            self.docked_widgets = []
            self.container_docked = False

    def close(self):
        """Close the widget and the docked widgets, if any"""
        self.remove_docked_widgets()
        self._viewer.window.remove_dock_widget(self)