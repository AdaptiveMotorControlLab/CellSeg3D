"""Basic napari plugin framework for inference and training."""
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import napari

# Qt
from qtpy.QtWidgets import QProgressBar, QSizePolicy

# local
from napari_cellseg3d import config, utils
from napari_cellseg3d import interface as ui
from napari_cellseg3d.code_plugins.plugin_base import BasePluginFolder

logger = utils.LOGGER


class ModelFramework(BasePluginFolder):
    """A framework with buttons to use for loading images, labels, models, etc. for both inference and training."""

    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
        parent=None,
        loads_images=True,
        loads_labels=True,
        has_results=True,
    ):
        """Creates a plugin framework with the following elements.

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
            sorted(self.available_models.keys()), text_label="Model name"
        )

        self.weights_filewidget = ui.FilePathWidget(
            "Weights path", self._load_weights_path, self
        )

        self.custom_weights_choice = ui.CheckBox(
            "Load custom weights", self._toggle_weights_path, self
        )

        available_devices = ["CPU"] + [
            f"GPU {i}" for i in range(torch.cuda.device_count())
        ]
        self.device_choice = ui.DropdownMenu(
            available_devices,
            parent=self,
            text_label="Device",
        )
        self.device_choice.tooltips = "Choose the device to use for training.\nIf you have a GPU, it is recommended to use it"
        self.device_choice.setCurrentIndex(len(available_devices) - 1)

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
        """Emit a signal to print in a Log."""
        if self.log is not None:
            self.log.print_and_log(text)

    def save_log(self, do_timestamp=True):
        """Saves the worker's log to disk at self.results_path when called."""
        if self.log is not None:
            log = self.log.toPlainText()

            path = self.results_path

            if do_timestamp:
                log_name = f"Log_report_{utils.get_date_time()}.txt"
            else:
                log_name = "Log_report.txt"

            if len(log) != 0:
                with Path.open(
                    Path(path) / log_name,
                    "x",
                ) as f:
                    f.write(log)
                    f.close()
            else:
                logger.warning(
                    "No job has been completed yet, please start one or re-open the log window."
                )
        else:
            logger.warning(f"No logger defined : Log is {self.log}")

    def save_log_to_path(self, path, do_timestamp=True):
        """Saves the worker log to a specific path. Cannot be used with connect.

        Args:
            path (str): path to save folder
            do_timestamp (bool, optional): whether to add a timestamp to the log name. Defaults to True.
        """
        log = self.log.toPlainText()

        if do_timestamp:
            log_name = f"Log_report_{utils.get_date_time()}.txt"
        else:
            log_name = "Log_report.txt"

        path = str(Path(path) / log_name)

        if len(log) != 0:
            with Path.open(
                Path(path),
                "x",
            ) as f:
                f.write(log)
                f.close()
        else:
            logger.warning(
                "No job has been completed yet, please start one or re-open the log window."
            )

    def display_status_report(self):
        """Adds a text log, a progress bar and a "save log" button on the left side of the viewer (usually when starting a worker)."""
        # if self.container_report is None or self.log is None:
        #     logger.warning(
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
        """Toggle visibility of weight path."""
        ui.toggle_visibility(
            self.custom_weights_choice, self.weights_filewidget
        )

    def get_unsupervised_image_filepaths(self):
        """Returns a list of filepaths to images in the unsupervised images folder."""
        volume_directory = Path(
            self.unsupervised_images_filewidget.text_field.text()
        ).resolve()
        logger.debug(f"Volume directory : {volume_directory}")
        return utils.get_all_matching_files(volume_directory)

    def create_dataset_dict_no_labs(self):
        """Creates unsupervised data dictionary for MONAI transforms and training."""
        images_filepaths = self.get_unsupervised_image_filepaths()
        if len(images_filepaths) == 0:
            raise ValueError(
                f"Data folder {self.unsupervised_images_filewidget.text_field.text()} is empty"
            )

        logger.info("Images :")
        for file in images_filepaths:
            logger.info(Path(file).stem)
        logger.info("*" * 10)

        return [{"image": str(image_name)} for image_name in images_filepaths]

    def create_train_dataset_dict(self):
        """Creates data dictionary for MONAI transforms and training.

        Returns:
            A dict with the following keys: "image", "label"
        """
        logger.debug(f"Images : {self.images_filepaths}")
        logger.debug(f"Labels : {self.labels_filepaths}")

        logger.debug(f"Images : {self.images_filepaths}")
        logger.debug(f"Labels : {self.labels_filepaths}")

        if len(self.images_filepaths) == 0 or len(self.labels_filepaths) == 0:
            raise ValueError("Data folders are empty")

        if not Path(self.images_filepaths[0]).parent.exists():
            raise ValueError("Images folder does not exist")
        if not Path(self.labels_filepaths[0]).parent.exists():
            raise ValueError("Labels folder does not exist")

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

    @staticmethod
    def get_available_models():
        """Getter for module (class and functions) associated to currently selected model."""
        return config.MODEL_LIST

    # def load_model_path(self): # TODO add custom models
    #     """Show file dialog to set :py:attr:`model_path`"""
    #     folder = ui.open_folder_dialog(self, self._default_folders)
    #     if folder is not None and type(folder) is str and os.path.isdir(folder):
    #         self.model_path = folder
    #         self.lbl_model_path.setText(self.model_path)
    #         # self.update_default()

    def _update_weights_path(self, file):
        if file[0] == self._default_weights_folder:
            return
        if file is not None and file[0] != "":
            self.weights_config.path = file[0]
            self.weights_filewidget.text_field.setText(file[0])
            self._default_weights_folder = str(Path(file[0]).parent)

    def _load_weights_path(self):
        """Show file dialog to set :py:attr:`model_path`."""
        # logger.debug(self._default_weights_folder)

        file = ui.open_file_dialog(
            self,
            [self._default_weights_folder],
            file_extension="Weights file (*.pth)",
        )
        self._update_weights_path(file)

    def check_device_choice(self):
        """Checks the device choice in the UI and returns the corresponding torch device."""
        choice = self.device_choice.currentText()
        if choice == "CPU":
            device = "cpu"
        elif "GPU" in choice:
            i = int(choice.split(" ")[1])
            device = f"cuda:{i}"
        else:
            device = self.get_device()
        logger.debug(f"DEVICE choice : {device}")
        return device

    @staticmethod
    def get_device(show=True):
        """Tries to use the device specified by user and uses it for tensor operations.

        If not available, automatically discovers any cuda device.
        If none is available (CUDA not installed), uses cpu instead.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if show:
            logger.info(f"Using {device} device")
            logger.info("Using torch :")
            logger.info(torch.__version__)
        return device

    def empty_cuda_cache(self):
        """Empties the cuda cache if the device is a cuda device."""
        if self.get_device(show=False).type == "cuda":
            logger.info("Emptying cache...")
            torch.cuda.empty_cache()
            logger.info("Attempt complete : Cache emptied")

    def _build(self):
        raise NotImplementedError("Should be defined in children classes")
