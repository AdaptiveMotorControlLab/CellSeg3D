import warnings
from pathlib import Path

import napari
import torch

# Qt
from qtpy.QtWidgets import QProgressBar
from qtpy.QtWidgets import QSizePolicy

# local
from napari_cellseg3d import config
from napari_cellseg3d import interface as ui
from napari_cellseg3d import utils
from napari_cellseg3d.code_plugins.plugin_base import BasePluginFolder

warnings.formatwarning = utils.format_Warning
logger = utils.LOGGER


class ModelFramework(BasePluginFolder):
    """A framework with buttons to use for loading images, labels, models, etc. for both inference and training"""

    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
        parent=None,
        loads_images=True,
        loads_labels=True,
        has_results=True,
    ):
        """Creates a plugin framework with the following elements :

        * A button to choose an image folder containing the images of a dataset (e.g. dataset/images)

        * A button to choose a label folder containing the labels of a dataset (e.g. dataset/labels)

        * A button to choose a results folder to save results in (e.g. dataset/inference_results)

        * A file extension choice to choose which file types to load in the data folders

        * A docked module with a log, a progress bar and a save button (see :py:func:`~display_status_report`)

        Args:
            viewer (napari.viewer.Viewer): viewer to load the widget in
            parent: parent QWidget
            loads_images: if True, will contain UI elements used to load napari image layers
            loads_labels: if True, will contain UI elements used to load napari label layers
            has_results: if True, will add UI to choose a results path
        """
        super().__init__(
            viewer, parent, loads_images, loads_labels, has_results
        )

        self._viewer = viewer
        """Viewer to display the widget in"""

        # self.model_path = "" # TODO add custom models
        # """str: path to custom model defined by user"""

        self.weights_config = config.WeightsInfo()
        """str : path to custom weights defined by user"""

        self._default_weights_folder = self.weights_config.path
        """Default path for plugin weights"""

        self.available_models = config.MODEL_LIST

        """dict: dictionary of available models, with string as key for name in widget display"""

        self.worker = None
        """Worker from model_workers.py, either inference or training"""

        #######################################################
        # interface

        # TODO : implement custom model
        # self.model_filewidget = ui.FilePathWidget(
        #     "Model path", self.load_model_path, self
        # )

        self.model_choice = ui.DropdownMenu(
            sorted(self.available_models.keys()), label="Model name"
        )

        self.weights_filewidget = ui.FilePathWidget(
            "Weights path", self._load_weights_path, self
        )

        self.custom_weights_choice = ui.CheckBox(
            "Load custom weights", self._toggle_weights_path, self
        )

        ###################################################
        # status report docked widget

        self.report_container = ui.ContainerWidget(l=10, t=5, r=5, b=5)

        self.report_container.setSizePolicy(
            QSizePolicy.Fixed, QSizePolicy.Minimum
        )
        self.container_docked = False  # check if already docked

        self.progress = QProgressBar(self.report_container)
        self.progress.setVisible(False)
        """Widget for the progress bar"""

        self.log = ui.Log(self.report_container)
        self.log.setVisible(False)
        """Read-only display for process-related info. Use only for info destined to user."""

        self.btn_save_log = ui.Button(
            "Save log in results folder",
            func=self.save_log,
            parent=self.report_container,
            fixed=False,
        )
        self.btn_save_log.setVisible(False)

    def send_log(self, text):
        """Emit a signal to print in a Log"""
        if self.log is not None:
            self.log.print_and_log(text)

    def save_log(self):
        """Saves the worker's log to disk at self.results_path when called"""
        if self.log is not None:
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
        else:
            warnings.warn(f"No logger defined : Log is {self.log}")

    def save_log_to_path(self, path):
        """Saves the worker log to a specific path. Cannot be used with connect.

        Args:
            path (str): path to save folder
        """

        log = self.log.toPlainText()
        path = str(
            Path(path) / Path(f"Log_report_{utils.get_date_time()}.txt")
        )

        if len(log) != 0:
            with open(
                path,
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
                self.report_container.layout,
                [self.progress, self.log, self.btn_save_log],
                alignment=None,
            )

            self.report_container.setLayout(self.report_container.layout)

            report_dock = self._viewer.window.add_dock_widget(
                self.report_container,
                name="Status report",
                area="left",
                allowed_areas=["left"],
            )
            report_dock._close_btn = False

            # TODO move to activity log once they figure out _qt_window access and private attrib.
            # activity_log = self._viewer.window._qt_window._activity_dialog
            # activity_layout = activity_log._activityLayout
            # activity_layout.addWidget(self.container_report)

            self.docked_widgets.append(report_dock)
            self.container_docked = True

        self.log.setVisible(True)
        self.progress.setVisible(True)
        self.btn_save_log.setVisible(True)
        self.progress.setValue(0)

    def _toggle_weights_path(self):
        """Toggle visibility of weight path"""
        ui.toggle_visibility(
            self.custom_weights_choice, self.weights_filewidget
        )

    def create_train_dataset_dict(self):
        """Creates data dictionary for MONAI transforms and training.

        Returns:
            A dict with the following keys

            * "image": image
            * "label" : corresponding label
        """

        if len(self.images_filepaths) == 0 or len(self.labels_filepaths) == 0:
            raise ValueError("Data folders are empty")

        logger.info("Images :\n")
        for file in self.images_filepaths:
            logger.info(Path(file).name)
        logger.info("*" * 10)
        logger.info("Labels :\n")
        for file in self.labels_filepaths:
            logger.info(Path(file).name)

        data_dicts = [
            {"image": image_name, "label": label_name}
            for image_name, label_name in zip(
                self.images_filepaths, self.labels_filepaths
            )
        ]
        logger.debug(f"Training data dict : {data_dicts}")

        return data_dicts

    def get_model(self, key):  # TODO remove
        """Getter for module (class and functions) associated to currently selected model"""
        return self.models_dict[key]

    @staticmethod
    def get_available_models():
        """Getter for module (class and functions) associated to currently selected model"""
        return config.MODEL_LIST

    # def load_model_path(self): # TODO add custom models
    #     """Show file dialog to set :py:attr:`model_path`"""
    #     folder = ui.open_folder_dialog(self, self._default_folders)
    #     if folder is not None and type(folder) is str and os.path.isdir(folder):
    #         self.model_path = folder
    #         self.lbl_model_path.setText(self.model_path)
    #         # self.update_default()

    def _load_weights_path(self):
        """Show file dialog to set :py:attr:`model_path`"""

        # logger.debug(self._default_weights_folder)

        file = ui.open_file_dialog(
            self,
            [self._default_weights_folder],
            filetype="Weights file (*.pth)",
        )
        if file[0] == self._default_weights_folder:
            return
        if file is not None:
            if file[0] != "":
                self.weights_config.path = file[0]
                self.weights_filewidget.text_field.setText(file[0])
                self._default_weights_folder = str(Path(file[0]).parent)

    @staticmethod
    def get_device(show=True):
        """Automatically discovers any cuda device and uses it for tensor operations.
        If none is available (CUDA not installed), uses cpu instead."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if show:
            logger.info(f"Using {device} device")
            logger.info("Using torch :")
            logger.info(torch.__version__)
        return device

    def empty_cuda_cache(self):
        """Empties the cuda cache if the device is a cuda device"""
        if self.get_device(show=False).type == "cuda":
            logger.info("Attempting to empty cache...")
            torch.cuda.empty_cache()
            logger.info("Attempt complete : Cache emptied")

    # def update_default(self): # TODO add custom models
    #     """Update default path for smoother file dialogs, here with :py:attr:`~model_path` included"""
    #
    #     if len(self.images_filepaths) != 0:
    #         from_images = str(Path(self.images_filepaths[0]).parent)
    #     else:
    #         from_images = None
    #
    #     if len(self.labels_filepaths) != 0:
    #         from_labels = str(Path(self.labels_filepaths[0]).parent)
    #     else:
    #         from_labels = None
    #
    #     possible_paths = [
    #         path
    #         for path in [
    #             from_images,
    #             from_labels,
    #             # self.model_path,
    #             self.results_path,
    #         ]
    #         if path is not None
    #     ]
    #     self._default_folders = possible_paths
    # update if model_path is used again

    def _build(self):
        raise NotImplementedError("Should be defined in children classes")
