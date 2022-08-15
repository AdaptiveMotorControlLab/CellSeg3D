import os
import warnings

import napari
import torch

# Qt
from qtpy.QtWidgets import QProgressBar
from qtpy.QtWidgets import QSizePolicy

# local
from napari_cellseg3d import interface as ui
from napari_cellseg3d import utils
from napari_cellseg3d.log_utility import Log
from napari_cellseg3d.models import model_SegResNet as SegResNet
from napari_cellseg3d.models import model_SwinUNetR as SwinUNetR

# from napari_cellseg3d.models import model_TRAILMAP as TRAILMAP
from napari_cellseg3d.models import model_VNet as VNet
from napari_cellseg3d.models import model_TRAILMAP_MS as TRAILMAP_MS
from napari_cellseg3d.plugin_base import BasePluginFolder

warnings.formatwarning = utils.format_Warning


class ModelFramework(BasePluginFolder):
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
        super().__init__(viewer)

        self._viewer = viewer
        """Viewer to display the widget in"""

        self.model_path = ""
        """str: path to custom model defined by user"""
        self.weights_path = ""
        """str : path to custom weights defined by user"""

        self._default_path = [
            self.images_filepaths,
            self.labels_filepaths,
            self.model_path,
            self.weights_path,
            self.results_path,
        ]
        """Update defaults from PluginBaseFolder with model_path"""

        self.models_dict = {
            "VNet": VNet,
            "SegResNet": SegResNet,
            # "TRAILMAP": TRAILMAP,
            "TRAILMAP_MS": TRAILMAP_MS,
            "SwinUNetR": SwinUNetR,
        }
        """dict: dictionary of available models, with string for widget display as key

        Currently implemented : SegResNet, VNet, TRAILMAP_MS"""

        self.worker = None
        """Worker from model_workers.py, either inference or training"""

        #######################################################
        # interface

        # TODO : implement custom model
        self.model_filewidget = ui.FilePathWidget(
            "Model path", self.load_model_path, self
        )
        self.btn_model_path = self.model_filewidget.get_button()
        self.lbl_model_path = self.model_filewidget.get_text_field()

        self.model_choice = ui.DropdownMenu(
            sorted(self.models_dict.keys()), label="Model name"
        )
        self.lbl_model_choice = self.model_choice.label

        self.weights_filewidget = ui.FilePathWidget(
            "Weights path", self.load_weights_path, self
        )
        self.btn_weights_path = self.weights_filewidget.get_button()
        self.lbl_weights_path = self.weights_filewidget.get_text_field()

        self.weights_path_container = ui.combine_blocks(
            self.btn_weights_path, self.lbl_weights_path, b=0
        )
        self.weights_path_container.setVisible(False)

        self.custom_weights_choice = ui.make_checkbox(
            "Load custom weights", self.toggle_weights_path, self
        )

        ###################################################
        # status report docked widget
        (
            self.container_report,
            self.container_report_layout,
        ) = ui.make_container(10, 5, 5, 5)
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

        self.btn_save_log = ui.Button(
            "Save log in results folder",
            func=self.save_log,
            parent=self.container_report,
            fixed=False,
        )
        self.btn_save_log.setVisible(False)
        #####################################################

    def send_log(self, text):
        """Emit a signal to print in a Log"""
        self.log.print_and_log(text)

    def save_log(self):
        """Saves the worker's log to disk at self.results_path when called"""
        log = self.log.toPlainText()

        path = self.results_path

        if len(log) != 0:
            with open(
                path + f"/Log_report_{utils.get_date_time()}.txt",
                "x",
            ) as f:
                f.write(log)
                f.close()
        else:
            warnings.warn(
                "No job has been completed yet, please start one or re-open the log window."
            )

    def save_log_to_path(self, path):
        """Saves the worker log to a specific path. Cannot be used with connect.

        Args:
            path (str): path to save folder
        """

        log = self.log.toPlainText()

        if len(log) != 0:
            with open(
                path + f"/Log_report_{utils.get_date_time()}.txt",
                "x",
            ) as f:
                f.write(log)
                f.close()
        else:
            warnings.warn(
                "No job has been completed yet, please start one or re-open the log window."
            )

    def display_status_report(self):
        """Adds a text log, a progress bar and a "save log" button on the left side of the viewer
        (usually when starting a worker)"""

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
        #     self.btn_save_log = ui.Button(
        #         "Save log in results folder", parent=self.container_report
        #     )
        #     self.btn_save_log.clicked.connect(self.save_log)
        #
        #     self.container_docked = False  # check if already docked

        if self.container_docked:
            self.log.clear()
        elif not self.container_docked:

            ui.add_widgets(
                self.container_report_layout,
                [self.progress, self.log, self.btn_save_log],
                alignment=None,
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

    def toggle_weights_path(self):
        """Toggle visibility of weight path"""
        ui.toggle_visibility(
            self.custom_weights_choice, self.weights_path_container
        )

    def create_train_dataset_dict(self):
        """Creates data dictionary for MONAI transforms and training.

        Returns: a dict with the following :
            **Keys:**

            * "image": image

            * "label" : corresponding label
        """

        if len(self.images_filepaths) == 0 or len(self.labels_filepaths) == 0:
            raise ValueError("Data folders are empty")

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

    def load_model_path(self):
        """Show file dialog to set :py:attr:`model_path`"""
        dir = ui.open_file_dialog(self, self._default_path)
        if dir != "" and type(dir) is str and os.path.isdir(dir):
            self.model_path = dir
            self.lbl_model_path.setText(self.results_path)
            # self.update_default()

    def load_weights_path(self):
        """Show file dialog to set :py:attr:`model_path`"""
        file = ui.open_file_dialog(
            self, self._default_path, filetype="Weights file (*.pth)"
        )
        if file != "":
            self.weights_path = file[0]
            self.lbl_weights_path.setText(self.weights_path)
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

    def update_default(self):
        """Update default path for smoother file dialogs, here with :py:attr:`~model_path` included"""
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

    def build(self):
        raise NotImplementedError("Should be defined in children classes")
