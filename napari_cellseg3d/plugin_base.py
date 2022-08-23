import glob
from pathlib import Path
import warnings

import napari
from qtpy.QtWidgets import QSizePolicy
from qtpy.QtWidgets import QTabWidget

from napari_cellseg3d import interface as ui


class BasePluginSingleImage(QTabWidget):
    """A basic plugin template for working with **single images**"""

    def __init__(self, viewer: "napari.viewer.Viewer", parent=None):
        """Creates a Base plugin with several buttons pre-defined but not added to a layout :

        * Open file prompt to select images directory

        * Open file prompt to select labels directory

        * A checkbox to choose whether to load a folder of images or a single 3D file
          If toggled, shows a filetype option to select the extension

        * A close button that closes the widget

        * A dropdown menu with a choice of png or tif filetypes
        """

        super().__init__()

        self.parent = parent
        """Parent widget"""
        self._viewer = viewer
        """napari.viewer.Viewer: viewer in which the widget is displayed"""

        self.docked_widgets = []
        self.container_docked = False

        self.image_path = None
        """str: path to image folder"""

        self.label_path = None
        """str: path to label folder"""

        self.results_path = None
        """str: path to results folder"""

        self.filetype = ""
        """str: filetype, .tif or .png"""
        self.as_folder = False
        """bool: Whether to load a single file or a folder as a stack"""

        self._default_path = [self.image_path, self.label_path]

        self.image_filewidget = ui.FilePathWidget(
            "Image path", self.show_dialog_images, self
        )
        self.btn_image = self.image_filewidget.get_button()
        """Button to load image folder"""
        self.lbl_image = self.image_filewidget.get_text_field()

        self.label_filewidget = ui.FilePathWidget(
            "Label path", self.show_dialog_labels, self
        )
        self.lbl_label = self.label_filewidget.get_text_field()
        self.btn_label = self.label_filewidget.get_button()
        """Button to load label folder"""

        self.filetype_choice = ui.DropdownMenu([".png", ".tif"])

        self.load_stack_choice = ui.CheckBox(
            "Load folder as stack ?", self.show_filetype_choice
        )
        """Checkbox to choose single file or directory loader handling"""

        self.load_stack_choice.setSizePolicy(
            QSizePolicy.Fixed, QSizePolicy.Fixed
        )

        self.btn_close = ui.Button("Close", self.remove_from_viewer, self)

        # self.lbl_ft = QLabel("Filetype :", self)
        # self.lbl_ft2 = QLabel("(Folders of .png or single .tif files)", self)

    def build(self):
        """Method to be defined by children classes"""
        raise NotImplementedError

    def show_filetype_choice(self):
        """Method to show/hide the filetype choice when "loading as folder" is (de)selected"""
        show = self.load_stack_choice.isChecked()
        if show is not None:
            self.filetype_choice.setVisible(show)
            # self.lbl_ft.setVisible(show)

    def show_file_dialog(self):
        """Open file dialog and process path depending on single file/folder loading behaviour"""
        if self.load_stack_choice.isChecked():
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

    def show_dialog_images(self):
        """Show file dialog and set image path"""
        f_name = self.show_file_dialog()
        if type(f_name) is str and f_name != "":
            self.image_path = f_name
            self.lbl_image.setText(self.image_path)
            self.update_default()

    def show_dialog_labels(self):
        """Show file dialog and set label path"""
        f_name = self.show_file_dialog()
        if type(f_name) is str and f_name != "":
            self.label_path = f_name
            self.lbl_label.setText(self.label_path)
            self.update_default()

    def load_results_path(self):
        """Show file dialog to set :py:attr:`~results_path`"""
        folder = ui.open_folder_dialog(self, self._default_path)
        if folder != "" and type(folder) is folder and Path(folder).is_dir():
            self.results_path = folder
            self.lbl_result_path.setText(self.results_path)
            self.update_default()

    def update_default(self):
        """Updates default path for smoother navigation when opening file dialogs"""
        self._default_path = [self.image_path, self.label_path]

    def remove_from_viewer(self):
        """Removes the widget from the napari window.
        Can be re-implemented in children classes if needed"""

        self.remove_docked_widgets()

        if self.parent is not None:
            self.parent.remove_from_viewer()  # TODO keep this way ?
            return
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
            return True
        except LookupError:
            return False


class BasePluginFolder(QTabWidget):
    """A basic plugin template for working with **folders of images**"""

    def __init__(self, viewer: "napari.viewer.Viewer", parent=None):
        """Creates a plugin template with the following widgets defined but not added in a layout :

        * A button to load a folder of images

        * A button to load a folder of labels

        * A button to set a results folder

        * A dropdown menu to select the file extension to be loaded from the folders"""
        super().__init__()
        self.parent = parent
        self._viewer = viewer
        """Viewer to display the widget in"""

        self.images_filepaths = []
        """array(str): paths to images for training or inference"""
        self.labels_filepaths = []
        """array(str): paths to labels for training"""
        self.results_path = None
        """str: path to output folder,to save results in"""

        self._default_folder = [None]
        """Update defaults from PluginBaseFolder with model_path"""

        self.docked_widgets = []
        """List of docked widgets (returned by :py:func:`viewer.window.add_dock_widget())`,
        can be used to remove docked widgets"""

        #######################################################
        # interface
        self.image_filewidget = ui.FilePathWidget(
            "Images directory", self.load_image_dataset, self
        )
        self.btn_image_files = self.image_filewidget.get_button()
        self.lbl_image_files = self.image_filewidget.get_text_field()

        self.label_filewidget = ui.FilePathWidget(
            "Labels directory", self.load_label_dataset, self
        )
        self.btn_label_files = self.label_filewidget.get_button()
        self.lbl_label_files = self.label_filewidget.get_text_field()

        self.filetype_choice = ui.DropdownMenu(
            [".tif", ".tiff"], label="File format"
        )
        self.lbl_filetype = self.filetype_choice.label
        """Allows to choose which file will be loaded from folder"""

        self.results_filewidget = ui.FilePathWidget(
            "Results directory", self.load_results_path, self
        )
        self.btn_result_path = self.results_filewidget.get_button()
        self.lbl_result_path = self.results_filewidget.get_text_field()
        #######################################################

    def make_close_button(self):
        btn = ui.Button("Close", self.remove_from_viewer)
        btn.setToolTip(
            "Close the window and all docked widgets. Make sure to save your work !"
        )
        return btn

    def make_prev_button(self):
        btn = ui.Button(
            "Previous", lambda: self.setCurrentIndex(self.currentIndex() - 1)
        )
        return btn

    def make_next_button(self):
        btn = ui.Button(
            "Next", lambda: self.setCurrentIndex(self.currentIndex() + 1)
        )
        return btn

    def load_dataset_paths(self):
        """Loads all image paths (as str) in a given folder for which the extension matches the set filetype

        Returns:
           array(str): all loaded file paths
        """
        filetype = self.filetype_choice.currentText()
        directory = ui.open_folder_dialog(self, self._default_folder)

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

        if filenames:
            self.images_filepaths = sorted(filenames)

            path = str(Path(filenames[0]).parent)
            self.lbl_image_files.setText(path)
            self.update_default()

    def load_label_dataset(self):
        """Show file dialog to set :py:attr:`~labels_filepaths`"""
        filenames = self.load_dataset_paths()
        if filenames:
            self.labels_filepaths = sorted(filenames)
            path = str(Path(filenames[0]).parent)
            self.lbl_label_files.setText(path)
            self.update_default()

    def load_results_path(self):
        """Show file dialog to set :py:attr:`~results_path`"""
        folder = ui.open_folder_dialog(self, self._default_folder)
        if folder != "" and type(folder) is str and Path(folder).is_dir():
            self.results_path = folder
            # print(f"Results path : {self.results_path}")
            self.lbl_result_path.setText(self.results_path)
            self.update_default()

    def build(self):
        raise NotImplementedError("Should be defined in children classes")

    def update_default(self):
        """Update default path for smoother file dialogs"""
        if len(self.images_filepaths) != 0:
            from_images = str(Path(self.images_filepaths[0]).parent)
        else:
            from_images = None

        if len(self.labels_filepaths) != 0:
            from_labels = str(Path(self.labels_filepaths[0]).parent)
        else:
            from_labels = None

        self._default_folder = [
            path
            for path in [
                from_images,
                from_labels,
                self.results_path,
            ]
            if (path != [] and path is not None)
        ]

    def remove_docked_widgets(self):
        """Removes docked widgets and resets checks for status report"""
        if len(self.docked_widgets) != 0:
            [
                self._viewer.window.remove_dock_widget(w)
                for w in self.docked_widgets
                if w is not None
            ]
            self.docked_widgets = []
            self.container_docked = False

    def remove_from_viewer(self):
        """Close the widget and the docked widgets, if any"""
        self.remove_docked_widgets()
        if self.parent is not None:
            self.parent.remove_from_viewer()
            return
        self._viewer.window.remove_dock_widget(self)
