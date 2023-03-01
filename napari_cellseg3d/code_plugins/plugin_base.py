import warnings
from functools import partial
from pathlib import Path

import napari

# Qt
from qtpy.QtCore import qInstallMessageHandler
from qtpy.QtWidgets import QTabWidget
from qtpy.QtWidgets import QWidget

# local
from napari_cellseg3d import interface as ui
from napari_cellseg3d import utils

logger = utils.LOGGER


class BasePluginSingleImage(QTabWidget):
    """A basic plugin template for working with **single images**"""

    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
        parent=None,
        loads_images=True,
        loads_labels=True,
        has_results=True,
    ):
        """
        Creates a Base plugin with several buttons pre-defined

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
        self.show_image_io = loads_images

        self.label_path = None
        """str: path to label folder"""
        self.show_label_io = loads_labels

        self.results_path = None
        """str: path to results folder"""
        self.show_results_io = has_results

        self._default_path = [self.image_path, self.label_path]

        ################
        self.layer_choice = ui.RadioButton("Layer", parent=self)
        self.folder_choice = ui.RadioButton("Folder", parent=self)
        self.radio_buttons = ui.combine_blocks(
            self.folder_choice, self.layer_choice
        )
        self.io_panel = None  # call self._build_io_panel to build
        ################
        # Image widgets
        self.image_filewidget = ui.FilePathWidget(
            "Image path", self._show_dialog_images, self
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
            "Label path", self._show_dialog_labels, parent=self
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

        self.filetype_choice = ui.DropdownMenu(
            [".tif", ".tiff"], label="File format"
        )
        ########
        qInstallMessageHandler(ui.handle_adjust_errors_wrapper(self))

    def enable_utils_menu(self):
        """
        Enables the usage of the CTRL+right-click shortcut to the utilities.
        Should only be used in "high-level" widgets (provided in napari Plugins menu) to avoid multiple activation
        """
        viewer = self._viewer

        @viewer.mouse_drag_callbacks.append
        def show_menu(_, event):
            return ui.UtilsDropdown().dropdown_menu_call(self, event)

    def _build_io_panel(self):
        self.io_panel = ui.GroupedWidget("Data")

        # self.io_panel.setToolTip("IO Panel")

        ui.add_widgets(
            self.io_panel.layout,
            [
                self.radio_buttons,
                self.image_layer_loader,
                self.label_layer_loader,
                self.filetype_choice,
                self.image_filewidget,
                self.labels_filewidget,
                self.results_filewidget,
            ],
        )
        self.io_panel.setLayout(self.io_panel.layout)

        # self._set_io_visibility()
        return self.io_panel

    def _remove_unused(self):
        if not self.show_label_io:
            self.labels_filewidget = None
            self.label_layer_loader = None

        if not self.show_image_io:
            self.image_layer_loader = None
            self.image_filewidget = None

        if not self.show_results_io:
            self.results_filewidget = None

    def _set_io_visibility(self):
        ##################
        # Show when layer is selected
        if self.show_image_io:
            self._show_io_element(self.image_layer_loader, self.layer_choice)
        else:
            self._hide_io_element(self.image_layer_loader)
        if self.show_label_io:
            self._show_io_element(self.label_layer_loader, self.layer_choice)
        else:
            self._hide_io_element(self.label_layer_loader)

        ##################
        # Show when folder is selected
        f = self.folder_choice

        self._show_io_element(self.filetype_choice, f)
        if self.show_image_io:
            self._show_io_element(self.image_filewidget, f)
        else:
            self._hide_io_element(self.image_filewidget)
        if self.show_label_io:
            self._show_io_element(self.labels_filewidget, f)
        else:
            self._hide_io_element(self.labels_filewidget)
        if not self.show_results_io:
            self._hide_io_element(self.results_filewidget)

        self.folder_choice.toggle()
        self.layer_choice.toggle()

    @staticmethod
    def _show_io_element(widget: QWidget, toggle: QWidget = None):
        """
        Args:
            widget: Widget to be shown or hidden
            toggle: Toggle to be used to determine whether widget should be shown (Checkbox or RadioButton)
        """
        widget.setVisible(True)

        if toggle is not None:
            toggle.toggled.connect(
                partial(ui.toggle_visibility, toggle, widget)
            )

    @staticmethod
    def _hide_io_element(widget: QWidget, toggle: QWidget = None):
        """
        Attempts to disconnect widget from toggle and hide it.
        Args:
            widget: Widget to be hidden
            toggle: Toggle to be disconnected from widget, if any
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
        """Method to be defined by children classes"""
        raise NotImplementedError("To be defined in child classes")

    def _show_filetype_choice(self):
        """Method to show/hide the filetype choice when "loading as folder" is (de)selected"""
        show = self.load_as_stack_choice.isChecked()
        if show is not None:
            self.filetype_choice.setVisible(show)
            # self.lbl_ft.setVisible(show)

    def _show_file_dialog(self):
        """Open file dialog and process path depending on single file/folder loading behaviour"""
        if self.load_as_stack_choice.isChecked():
            folder = ui.open_folder_dialog(
                self,
                self._default_path,
                filetype=f"Image file (*{self.filetype_choice.currentText()})",
            )
            return folder
        else:
            f_name = ui.open_file_dialog(self, self._default_path)
            f_name = str(f_name[0])
            self.filetype = str(Path(f_name).suffix)
            return f_name

    def _show_dialog_images(self):
        """Show file dialog and set image path"""
        f_name = self._show_file_dialog()
        if type(f_name) is str and f_name != "":
            self.image_path = f_name
            self.image_filewidget.text_field.setText(self.image_path)
            self._update_default()

    def _show_dialog_labels(self):
        """Show file dialog and set label path"""
        f_name = self._show_file_dialog()
        if isinstance(f_name, str) and f_name != "":
            self.label_path = f_name
            self.labels_filewidget.text_field.setText(self.label_path)
            self._update_default()

    def _check_results_path(self, folder):
        if folder != "" and isinstance(folder, str):
            if not Path(folder).is_dir():
                Path(folder).mkdir(parents=True, exist_ok=True)
                if not Path(folder).is_dir():
                    return False
                logger.info(f"Created missing results folder : {folder}")
            return True
        return False

    def _load_results_path(self):
        """Show file dialog to set :py:attr:`~results_path`"""
        folder = ui.open_folder_dialog(self, self._default_path)

        if self._check_results_path(folder):
            self.results_path = folder
            # logger.debug(f"Results path : {self.results_path}")
            self.results_filewidget.text_field.setText(self.results_path)
            self._update_default()

    def _update_default(self):
        """Updates default path for smoother navigation when opening file dialogs"""
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
        btn = ui.Button(
            "Previous", lambda: self.setCurrentIndex(self.currentIndex() - 1)
        )
        return btn

    def _make_next_button(self):
        btn = ui.Button(
            "Next", lambda: self.setCurrentIndex(self.currentIndex() + 1)
        )
        return btn

    def remove_from_viewer(self):
        """Removes the widget from the napari window.
        Can be re-implemented in children classes if needed"""

        self.remove_docked_widgets()
        self._viewer.window.remove_dock_widget(self)

    def remove_docked_widgets(self):
        """Removes all docked widgets from napari window"""
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
    """A basic plugin template for working with **folders of images**"""

    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
        parent=None,
        loads_images=True,
        loads_labels=True,
        has_results=True,
    ):
        """Creates a plugin template with the following widgets defined but not added in a layout :

        * A button to load a folder of images

        * A button to load a folder of labels

        * A button to set a results folder

        * A dropdown menu to select the file extension to be loaded from the folders"""
        super().__init__(
            viewer, parent, loads_images, loads_labels, has_results
        )

        self.images_filepaths = []
        """array(str): paths to images for training or inference"""
        self.labels_filepaths = []
        """array(str): paths to labels for training"""
        self.results_path = None
        """str: path to output folder,to save results in"""

        self._default_folders = [None]
        """Update defaults from PluginBaseFolder with model_path"""

        self.docked_widgets = []
        """List of docked widgets (returned by :py:func:`viewer.window.add_dock_widget())`,
        can be used to remove docked widgets"""

        #######################################################
        # interface
        # self.image_filewidget = ui.FilePathWidget(
        #     "Images directory", self.load_image_dataset, self
        # )
        self.image_filewidget.text_field = "Images directory"
        self.image_filewidget.button.clicked.disconnect(
            self._show_dialog_images
        )
        self.image_filewidget.button.clicked.connect(self.load_image_dataset)

        # self.labels_filewidget = ui.FilePathWidget(
        #     "Labels directory", self.load_label_dataset, self
        # )
        self.labels_filewidget.text_field = "Labels directory"
        self.labels_filewidget.button.clicked.disconnect(
            self._show_dialog_labels
        )
        self.labels_filewidget.button.clicked.connect(self.load_label_dataset)

        # self.filetype_choice = ui.DropdownMenu(
        #     [".tif", ".tiff"], label="File format"
        # )
        """Allows to choose which file will be loaded from folder"""
        #######################################################
        # self._set_io_visibility()

    def load_dataset_paths(self):
        """Loads all image paths (as str) in a given folder for which the extension matches the set filetype

        Returns:
           array(str): all loaded file paths
        """
        filetype = self.filetype_choice.currentText()
        directory = ui.open_folder_dialog(self, self._default_folders)

        file_paths = sorted(Path(directory).glob("*" + filetype))
        if len(file_paths) == 0:
            warnings.warn(
                f"The folder does not contain any compatible {filetype} files.\n"
                f"Please check the validity of the folder and images."
            )

        return file_paths

    def load_image_dataset(self):
        """Show file dialog to set :py:attr:`~images_filepaths`"""
        filenames = self.load_dataset_paths()
        logger.debug(f"image filenames : {filenames}")
        if filenames:
            self.images_filepaths = [str(path) for path in sorted(filenames)]
            path = str(Path(filenames[0]).parent)
            self.image_filewidget.text_field.setText(path)
            self.image_filewidget.check_ready()
            self._update_default()

    def load_label_dataset(self):
        """Show file dialog to set :py:attr:`~labels_filepaths`"""
        filenames = self.load_dataset_paths()
        logger.debug(f"labels filenames : {filenames}")
        if filenames:
            self.labels_filepaths = [str(path) for path in sorted(filenames)]
            path = str(Path(filenames[0]).parent)
            self.labels_filewidget.text_field.setText(path)
            self.labels_filewidget.check_ready()
            self._update_default()

    def _update_default(self):
        """Update default path for smoother file dialogs"""
        if len(self.images_filepaths) != 0:
            from_images = str(Path(self.images_filepaths[0]).parent)
        else:
            from_images = None

        if len(self.labels_filepaths) != 0:
            from_labels = str(Path(self.labels_filepaths[0]).parent)
        else:
            from_labels = None

        self._default_folders = [
            path
            for path in [
                from_images,
                from_labels,
                self.results_path,
            ]
            if (path != [] and path is not None)
        ]
