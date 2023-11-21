"""Base classes for napari_cellseg3d plugins."""
from functools import partial
from pathlib import Path

import napari

# Qt
from qtpy.QtCore import qInstallMessageHandler
from qtpy.QtWidgets import QTabWidget, QWidget

# local
from napari_cellseg3d import interface as ui
from napari_cellseg3d import utils

logger = utils.LOGGER


class BasePluginSingleImage(QTabWidget):
    """A basic plugin template for working with **single images**."""

    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
        parent=None,
        loads_images=True,
        loads_labels=True,
        has_results=True,
    ):
        """Creates a Base plugin with several buttons pre-defined.

        Args:
            viewer: napari viewer to display in
            parent: parent QWidget. Defaults to None
            loads_images: whether to show image IO widgets
            loads_labels: whether to show labels IO widgets
            has_results: whether to show results IO widgets

        """
        super().__init__(parent)
        """Parent widget"""
        self._viewer = viewer
        """napari.viewer.Viewer: viewer in which the widget is displayed"""

        self.docked_widgets = []
        self.container_docked = False

        self.image_path = None
        """str: path to image folder"""
        self._show_image_io = loads_images

        self.label_path = None
        """str: path to label folder"""
        self._show_label_io = loads_labels

        self.results_path = None
        """str: path to results folder"""
        self._show_results_io = has_results

        self._default_path = [self.image_path, self.label_path]

        ################
        self.layer_choice = ui.RadioButton("Layer", parent=self)
        self.folder_choice = ui.RadioButton("Folder", parent=self)
        self.filetype = None
        self.radio_buttons = ui.combine_blocks(
            self.folder_choice, self.layer_choice
        )
        self.io_panel = None  # call self._build_io_panel to build
        ################
        # Image widgets
        self.image_filewidget = ui.FilePathWidget(
            "Image path",
            self._show_dialog_images,
            self,
            required=True,
        )

        self.image_layer_loader: ui.LayerSelecter = ui.LayerSelecter(
            self._viewer,
            name="Image :",
            layer_type=napari.layers.Image,
            parent=self,
        )
        """LayerSelecter for images"""
        ################
        # Label widgets
        self.labels_filewidget = ui.FilePathWidget(
            "Label path",
            self._show_dialog_labels,
            parent=self,
            required=True,
        )

        self.label_layer_loader: ui.LayerSelecter = ui.LayerSelecter(
            self._viewer,
            name="Labels :",
            layer_type=napari.layers.Labels,
            parent=self,
        )
        """LayerSelecter for labels"""
        ################
        # Results widget
        self.results_filewidget = ui.FilePathWidget(
            "Saving path", self._load_results_path, parent=self
        )
        ########
        qInstallMessageHandler(ui.handle_adjust_errors_wrapper(self))

    def enable_utils_menu(self):
        """Enables the usage of the CTRL+right-click shortcut to the utilities.

        Should only be used in "high-level" widgets (provided in napari Plugins menu) to avoid multiple activation.
        """
        viewer = self._viewer

        @viewer.mouse_drag_callbacks.append
        def show_menu(_, event):
            return ui.UtilsDropdown().dropdown_menu_call(self, event)

    def _build_io_panel(self):
        self.io_panel = ui.GroupedWidget("Data")
        self.save_label = ui.make_label("Save location :", parent=self)
        # self.io_panel.setToolTip("IO Panel")

        ui.add_widgets(
            self.io_panel.layout,
            [
                self.radio_buttons,
                self.image_layer_loader,
                self.label_layer_loader,
                # self.filetype_choice,
                self.image_filewidget,
                self.labels_filewidget,
                self.save_label,
                self.results_filewidget,
            ],
        )
        self.io_panel.setLayout(self.io_panel.layout)

        # self._set_io_visibility()
        return self.io_panel

    def _remove_unused(self):
        if not self._show_label_io:
            self.labels_filewidget = None
            self.label_layer_loader = None

        if not self._show_image_io:
            self.image_layer_loader = None
            self.image_filewidget = None

        if not self._show_results_io:
            self.results_filewidget = None

    def _set_io_visibility(self):
        ##################
        # Show when layer is selected
        if self._show_image_io:
            self._show_io_element(self.image_layer_loader, self.layer_choice)
        else:
            self._hide_io_element(self.image_layer_loader)
        if self._show_label_io:
            self._show_io_element(self.label_layer_loader, self.layer_choice)
        else:
            self._hide_io_element(self.label_layer_loader)

        ##################
        # Show when folder is selected
        f = self.folder_choice

        # self._show_io_element(self.filetype_choice, f)
        if self._show_image_io:
            self._show_io_element(self.image_filewidget, f)
        else:
            self._hide_io_element(self.image_filewidget)
        if self._show_label_io:
            self._show_io_element(self.labels_filewidget, f)
        else:
            self._hide_io_element(self.labels_filewidget)
        if not self._show_results_io:
            self._hide_io_element(self.results_filewidget)

        self.folder_choice.toggle()
        self.layer_choice.toggle()

    @staticmethod
    def _show_io_element(widget: QWidget, toggle: QWidget = None):
        """Show widget and connect it to toggle if any.

        Args:
            widget: Widget to be shown or hidden
            toggle: Toggle to be used to determine whether widget should be shown (Checkbox or RadioButton).
        """
        widget.setVisible(True)

        if toggle is not None:
            toggle.toggled.connect(
                partial(ui.toggle_visibility, toggle, widget)
            )

    @staticmethod
    def _hide_io_element(widget: QWidget, toggle: QWidget = None):
        """Attempts to disconnect widget from toggle and hide it.

        Args:
            widget: Widget to be hidden
            toggle: Toggle to be disconnected from widget, if any.
        """
        if toggle is not None:
            try:
                toggle.toggled.disconnect()
            except TypeError:
                logger.warning(
                    "Warning: no method was found to disconnect from widget visibility"
                )

        widget.setVisible(False)

    def _build(self):
        """Method to be defined by children classes."""
        raise NotImplementedError("To be defined in child classes")

    def _show_file_dialog(self):
        """Open file dialog and process path for a single file."""
        # if self.load_as_stack_choice.isChecked():
        # return ui.open_folder_dialog(
        #     self,
        #     self._default_path,
        #     filetype=f"Image file (*{self.filetype_choice.currentText()})",
        # )
        # else:
        logger.debug("Opening file dialog")
        f_name = ui.open_file_dialog(self, self._default_path)
        logger.debug(f"File dialog returned {f_name}")
        choice = str(f_name[0])
        self.filetype = str(Path(choice).suffix)
        logger.debug(f"Filetype set to {self.filetype}")
        self._update_default_paths()
        return choice

    def _show_dialog_images(self):
        """Show file dialog and set image path."""
        f_name = self._show_file_dialog()
        if type(f_name) is str and Path(f_name).is_file():
            self.image_path = f_name
            logger.debug(f"Image path set to {self.image_path}")
            self.image_filewidget.text_field.setText(self.image_path)
            self._update_default_paths()

    def _show_dialog_labels(self):
        """Show file dialog and set label path."""
        f_name = self._show_file_dialog()
        if isinstance(f_name, str) and Path(f_name).is_file():
            self.label_path = f_name
            logger.debug(f"Label path set to {self.label_path}")
            self.labels_filewidget.text_field.setText(self.label_path)
            self._update_default_paths()

    def _check_results_path(self, folder: str):
        """Check if results folder exists, create it if not."""
        logger.debug(f"Checking results folder : {folder}")
        if folder != "" and isinstance(folder, str):
            if not Path(folder).is_dir():
                Path(folder).mkdir(parents=True, exist_ok=True)
                if not Path(folder).is_dir():
                    logger.info(
                        f"Could not create missing results folder : {folder}"
                    )
                    return False
                logger.info(f"Created missing results folder : {folder}")
            return True
        if not isinstance(folder, str):
            raise TypeError(f"Expected string, got {type(folder)}")
        return False

    def _load_results_path(self):
        """Show file dialog to set :py:attr:`~results_path`."""
        self._update_default_paths()
        folder = ui.open_folder_dialog(self, self._default_path)

        if self._check_results_path(folder):
            self.results_path = str(Path(folder).resolve())
            logger.debug(f"Results path : {self.results_path}")
            self.results_filewidget.text_field.setText(self.results_path)
            self._update_default_paths()

    def _update_default_paths(self):
        """Updates default path for smoother navigation when opening file dialogs."""
        self._default_path = [
            self.image_path,
            self.label_path,
            self.results_path,
        ]

    def _make_close_button(self):
        btn = ui.Button("Close", self.remove_from_viewer)
        btn.setToolTip(
            "Close the window and all docked widgets. Make sure to save your work !"
        )
        return btn

    def _make_prev_button(self):
        return ui.Button(
            "Previous", lambda: self.setCurrentIndex(self.currentIndex() - 1)
        )

    def _make_next_button(self):
        return ui.Button(
            "Next", lambda: self.setCurrentIndex(self.currentIndex() + 1)
        )

    def remove_from_viewer(self):
        """Removes the widget from the napari window.

        Must be re-implemented in children classes where needed.
        """
        self.remove_docked_widgets()
        self._viewer.window.remove_dock_widget(self)

    def remove_docked_widgets(self):
        """Removes all docked widgets from napari window."""
        try:
            if len(self.docked_widgets) != 0:
                [
                    self._viewer.window.remove_dock_widget(w)
                    for w in self.docked_widgets
                    if w is not None
                ]
            self.docked_widgets = []
            self.container_docked = False
            return True
        except LookupError:
            return False


class BasePluginFolder(BasePluginSingleImage):
    """A basic plugin template for working with **folders of images**."""

    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
        parent=None,
        loads_images=True,
        loads_labels=True,
        has_results=True,
    ):
        """Creates a plugin template with the following widgets defined but not added in a layout.

        * A button to load a folder of images

        * A button to load a folder of labels

        * A button to set a results folder

        * A dropdown menu to select the file extension to be loaded from the folders
        """
        super().__init__(
            viewer, parent, loads_images, loads_labels, has_results
        )

        self.images_filepaths = []
        """array(str): paths to images for training or inference"""
        self.labels_filepaths = []
        """array(str): paths to labels for training"""
        self.validation_filepaths = []
        """array(str): paths to validation files (unsup. learning)"""
        self.results_path = None
        """str: path to output folder,to save results in"""

        self._default_path = [None]
        """Update defaults from PluginBaseFolder with model_path"""

        self.docked_widgets = []
        """List of docked widgets (returned by :py:func:`viewer.window.add_dock_widget())`,
        can be used to remove docked widgets"""

        #######################################################
        # interface
        self.image_filewidget.text_field = "Images directory"
        self.image_filewidget.button.clicked.disconnect(
            self._show_dialog_images
        )
        self.image_filewidget.button.clicked.connect(self.load_image_dataset)

        self.labels_filewidget.text_field = "Labels directory"
        self.labels_filewidget.button.clicked.disconnect(
            self._show_dialog_labels
        )
        self.labels_filewidget.button.clicked.connect(self.load_label_dataset)
        ################
        # Validation images widget
        self.unsupervised_images_filewidget = ui.FilePathWidget(
            description="Training directory",
            file_function=self.load_unsup_images_dataset,
            parent=self,
        )
        self.unsupervised_images_filewidget.setVisible(False)
        # self.filetype_choice = ui.DropdownMenu(
        #     [".tif", ".tiff"], label="File format"
        # )
        """Allows to choose which file will be loaded from folder"""
        #######################################################
        # self._set_io_visibility()

    def load_dataset_paths(self):
        """Loads all image paths (as str) in a given folder for which the extension matches the set filetype.

        Returns:
           array(str): all loaded file paths
        """
        # filetype = self.filetype_choice.currentText()
        directory = ui.open_folder_dialog(self, self._default_path)

        file_paths = utils.get_all_matching_files(directory)
        if len(file_paths) == 0:
            logger.warning(
                "The folder does not contain any compatible .tif files.\n"
                "Please check the validity of the folder and images."
            )

        return file_paths

    def load_image_dataset(self):
        """Show file dialog to set :py:attr:`~images_filepaths`."""
        filenames = self.load_dataset_paths()
        if filenames:
            logger.info("Images loaded :")
            for f in filenames:
                logger.info(f"{str(Path(f).name)}")
            self.images_filepaths = [str(path) for path in sorted(filenames)]
            path = str(Path(filenames[0]).parent)
            self.image_filewidget.text_field.setText(path)
            self.image_filewidget.check_ready()
            self._update_default_paths(path)

    def load_unsup_images_dataset(self):
        """Show file dialog to set :py:attr:`~val_images_filepaths`."""
        filenames = self.load_dataset_paths()
        if filenames:
            logger.info("Images loaded (unsupervised training) :")
            for f in filenames:
                logger.info(f"{str(Path(f).name)}")
            self.validation_filepaths = [
                str(path) for path in sorted(filenames)
            ]
            path = str(Path(filenames[0]).parent)
            self.unsupervised_images_filewidget.text_field.setText(path)
            self.unsupervised_images_filewidget.check_ready()
            self._update_default_paths(path)

    def load_label_dataset(self):
        """Show file dialog to set :py:attr:`~labels_filepaths`."""
        filenames = self.load_dataset_paths()
        if filenames:
            logger.info("Labels loaded :")
            for f in filenames:
                logger.info(f"{str(Path(f).name)}")
            self.labels_filepaths = [str(path) for path in sorted(filenames)]
            path = str(Path(filenames[0]).parent)
            self.labels_filewidget.text_field.setText(path)
            self.labels_filewidget.check_ready()
            self._update_default_paths(path)

    def _update_default_paths(self, path=None):
        """Update default path for smoother file dialogs."""
        logger.debug(f"Updating default paths with {path}")
        if path is None:
            self._default_path = [
                self.extract_dataset_paths(self.images_filepaths),
                self.extract_dataset_paths(self.labels_filepaths),
                self.extract_dataset_paths(self.validation_filepaths),
                self.results_path,
            ]
            return utils.parse_default_path(self._default_path)
        if Path(path).is_dir():
            self._default_path.append(path)
        return utils.parse_default_path(self._default_path)

    @staticmethod
    def extract_dataset_paths(paths):
        """Gets the parent folder name of the first image and label paths."""
        if len(paths) == 0:
            return None
        if paths[0] is None:
            return None
        return str(Path(paths[0]).parent)

    def _check_all_filepaths(self):
        self.image_filewidget.check_ready()
        self.labels_filewidget.check_ready()
        self.results_filewidget.check_ready()
        self.unsupervised_images_filewidget.check_ready()


class BasePluginUtils(BasePluginFolder):
    """Small subclass used to have centralized widgets layer and result path selection in utilities."""

    save_path = None
    utils_default_paths = [Path.home() / "cellseg3d"]

    def __init__(
        self,
        viewer: napari.viewer.Viewer,
        parent=None,
        loads_images=True,
        loads_labels=True,
    ):
        """Creates a plugin template with the following widgets defined but not added in a layout."""
        super().__init__(
            viewer=viewer,
            loads_images=loads_images,
            loads_labels=loads_labels,
            parent=parent,
        )
        if parent is not None:
            self.setParent(parent)
        self.parent = parent

        self.layer = None
        """Should contain the layer associated with the results of the utility widget"""

    def _update_default_paths(self, path=None):
        """Override to also update utilities' pool of default paths."""
        default_path = super()._update_default_paths(path)
        logger.debug(f"Trying to update default with {default_path}")
        if default_path is not None:
            self.utils_default_paths.append(default_path)
