import warnings
from pathlib import Path

import napari
import numpy as np
from qtpy.QtWidgets import QSizePolicy
from tifffile import imread
from tifffile import imwrite

import napari_cellseg3d.interface as ui
from napari_cellseg3d import utils
from napari_cellseg3d.code_models.model_instance_seg import clear_small_objects
from napari_cellseg3d.code_models.model_instance_seg import InstanceWidgets
from napari_cellseg3d.code_models.model_instance_seg import threshold
from napari_cellseg3d.code_models.model_instance_seg import to_semantic
from napari_cellseg3d.code_plugins.plugin_base import BasePluginFolder

# TODO break down into multiple mini-widgets
# TODO create parent class for utils modules to avoid duplicates

MAX_W = 200
MAX_H = 1000

logger = utils.LOGGER


def save_folder(results_path, folder_name, images, image_paths):
    """
    Saves a list of images in a folder

    Args:
        results_path: Path to the folder containing results
        folder_name: Name of the folder containing results
        images: List of images to save
        image_paths: list of filenames of images
    """
    results_folder = results_path / Path(folder_name)
    results_folder.mkdir(exist_ok=False, parents=True)

    for file, image in zip(image_paths, images):
        path = results_folder / Path(file).name

        imwrite(
            path,
            image,
        )
    logger.info(f"Saved processed folder as : {results_folder}")


def save_layer(results_path, image_name, image):
    """
    Saves an image layer at the specified path

    Args:
        results_path: path to folder containing result
        image_name: image name for saving
        image: data array containing image

    Returns:

    """
    path = str(results_path / Path(image_name))  # TODO flexible filetype
    logger.info(f"Saved as : {path}")
    imwrite(path, image)


def show_result(viewer, layer, image, name):
    """
    Adds layers to a viewer to show result to user

    Args:
        viewer: viewer to add layer in
        layer: type of the original layer the operation was run on, to determine whether it should be an Image or Labels layer
        image: the data array containing the image
        name: name of the added layer

    Returns:

    """
    if isinstance(layer, napari.layers.Image):
        logger.debug("Added resulting image layer")
        viewer.add_image(image, name=name)
    elif isinstance(layer, napari.layers.Labels):
        logger.debug("Added resulting label layer")
        viewer.add_labels(image, name=name)
    else:
        warnings.warn(
            f"Results not shown, unsupported layer type {type(layer)}"
        )


class AnisoUtils(BasePluginFolder):
    """Class to correct anisotropy in images"""

    def __init__(self, viewer: "napari.Viewer.viewer", parent=None):
        """
        Creates a AnisoUtils widget

        Args:
            viewer: viewer in which to process data
            parent: parent widget
        """
        super().__init__(
            viewer,
            parent,
            loads_labels=False,
        )

        self.data_panel = self._build_io_panel()

        self.image_layer_loader.layer_list.label.setText("Layer :")
        self.image_layer_loader.set_layer_type(napari.layers.Layer)

        self.aniso_widgets = ui.AnisotropyWidgets(self, always_visible=True)
        self.start_btn = ui.Button("Start", self._start)

        self.results_path = Path.home() / Path("cellseg3d/anisotropy")
        self.results_filewidget.text_field.setText(str(self.results_path))
        self.results_filewidget.check_ready()

        self._build()

    def _build(self):
        container = ui.ContainerWidget()

        ui.add_widgets(
            container.layout,
            [
                self.data_panel,
                self.aniso_widgets,
                self.start_btn,
            ],
        )

        ui.ScrollArea.make_scrollable(
            container.layout,
            self,
            max_wh=[MAX_W, MAX_H],  # , min_wh=[100, 200], base_wh=[100, 200]
        )

        self._set_io_visibility()
        self.setSizePolicy(
            QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding
        )

    def _start(self):
        self.results_path.mkdir(exist_ok=True, parents=True)
        zoom = self.aniso_widgets.scaling_zyx()

        if self.layer_choice.isChecked():
            if self.image_layer_loader.layer_data() is not None:
                layer = self.image_layer_loader.layer()

                data = np.array(layer.data)
                isotropic_image = utils.resize(data, zoom)

                save_layer(
                    self.results_path,
                    f"isotropic_{layer.name}_{utils.get_date_time()}.tif",
                    isotropic_image,
                )
                show_result(
                    self._viewer,
                    layer,
                    isotropic_image,
                    f"isotropic_{layer.name}",
                )

        elif self.folder_choice.isChecked():
            if len(self.images_filepaths) != 0:
                images = [
                    utils.resize(np.array(imread(file)), zoom)
                    for file in self.images_filepaths
                ]
                save_folder(
                    self.results_path,
                    f"isotropic_results_{utils.get_date_time()}",
                    images,
                    self.images_filepaths,
                )


class RemoveSmallUtils(BasePluginFolder):
    """
    Widget to remove small objects
    """

    def __init__(self, viewer: "napari.viewer.Viewer", parent=None):
        """
        Creates a RemoveSmallUtils widget

        Args:
            viewer: viewer in which to process data
            parent: parent widget
        """
        super().__init__(
            viewer,
            parent,
            loads_labels=False,
        )

        self.data_panel = self._build_io_panel()

        self.image_layer_loader.layer_list.label.setText("Layer :")
        self.image_layer_loader.set_layer_type(napari.layers.Layer)

        self.start_btn = ui.Button("Start", self._start)
        self.size_for_removal_counter = ui.IntIncrementCounter(
            lower=1,
            upper=100000,
            default=10,
            text_label="Remove all smaller than (pxs):",
        )

        self.results_path = Path.home() / Path("cellseg3d/small_removed")
        self.results_filewidget.text_field.setText(str(self.results_path))
        self.results_filewidget.check_ready()

        self.container = self._build()

        self.function = clear_small_objects

    def _build(self):
        container = ui.ContainerWidget()

        ui.add_widgets(
            self.data_panel.layout,
            [
                self.size_for_removal_counter.label,
                self.size_for_removal_counter,
                self.start_btn,
            ],
        )
        container.layout.addWidget(self.data_panel)

        ui.ScrollArea.make_scrollable(
            container.layout, self, max_wh=[MAX_W, MAX_H]
        )
        self._set_io_visibility()
        container.setSizePolicy(
            QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding
        )
        return container

    def _start(self):
        self.results_path.mkdir(exist_ok=True, parents=True)
        remove_size = self.size_for_removal_counter.value()

        if self.layer_choice:
            if self.image_layer_loader.layer_data() is not None:
                layer = self.image_layer_loader.layer()

                data = np.array(layer.data)
                removed = self.function(data, remove_size)

                save_layer(
                    self.results_path,
                    f"cleared_{layer.name}_{utils.get_date_time()}.tif",
                    removed,
                )
                show_result(
                    self._viewer, layer, removed, f"cleared_{layer.name}"
                )
        elif self.folder_choice.isChecked():
            if len(self.images_filepaths) != 0:
                images = [
                    clear_small_objects(file, remove_size, is_file_path=True)
                    for file in self.images_filepaths
                ]
                save_folder(
                    self.results_path,
                    f"small_removed_results_{utils.get_date_time()}",
                    images,
                    self.images_filepaths,
                )
        return


class ToSemanticUtils(BasePluginFolder):
    """
    Widget to create semantic labels from instance labels
    """

    def __init__(self, viewer: "napari.viewer.Viewer", parent=None):
        """
        Creates a ToSemanticUtils widget

        Args:
            viewer: viewer in which to process data
            parent: parent widget
        """
        super().__init__(
            viewer,
            parent,
            loads_images=False,
        )

        self.data_panel = self._build_io_panel()

        self.start_btn = ui.Button("Start", self._start)

        self.results_path = Path.home() / Path("cellseg3d/threshold")
        self.results_filewidget.text_field.setText(str(self.results_path))
        self.results_filewidget.check_ready()

        self._build()

    def _build(self):
        container = ui.ContainerWidget()

        ui.add_widgets(
            self.data_panel.layout,
            [
                self.start_btn,
            ],
        )
        container.layout.addWidget(self.data_panel)

        ui.ScrollArea.make_scrollable(
            container.layout, self, max_wh=[MAX_W, MAX_H]
        )
        self._set_io_visibility()
        container.setSizePolicy(
            QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding
        )

    def _start(self):
        Path(self.results_path).mkdir(exist_ok=True, parents=True)

        if self.layer_choice:
            if self.label_layer_loader.layer_data() is not None:
                layer = self.label_layer_loader.layer()

                data = np.array(layer.data)
                semantic = to_semantic(data)

                save_layer(
                    self.results_path,
                    f"semantic_{layer.name}_{utils.get_date_time()}.tif",
                    semantic,
                )
                show_result(
                    self._viewer, layer, semantic, f"semantic_{layer.name}"
                )
        elif self.folder_choice.isChecked():
            if len(self.images_filepaths) != 0:
                images = [
                    to_semantic(file, is_file_path=True)
                    for file in self.images_filepaths
                ]
                save_folder(
                    self.results_path,
                    f"semantic_results_{utils.get_date_time()}",
                    images,
                    self.images_filepaths,
                )


class ToInstanceUtils(BasePluginFolder):
    """
    Widget to convert semantic labels to instance labels
    """

    def __init__(self, viewer: "napari.viewer.Viewer", parent=None):
        """
        Creates a ToInstanceUtils widget

        Args:
            viewer: viewer in which to process data
            parent: parent widget
        """
        super().__init__(
            viewer,
            parent,
            loads_images=False,
        )

        self.data_panel = self._build_io_panel()
        self.label_layer_loader.set_layer_type(napari.layers.Layer)

        self.instance_widgets = InstanceWidgets(parent=self)

        self.start_btn = ui.Button("Start", self._start)

        self.results_path = Path.home() / Path("cellseg3d/instance")
        self.results_filewidget.text_field.setText(str(self.results_path))
        self.results_filewidget.check_ready()

        self._build()

    def _build(self):
        container = ui.ContainerWidget()

        ui.add_widgets(
            container.layout,
            [
                self.data_panel,
                self.instance_widgets,
            ],
        )

        ui.add_widgets(self.instance_widgets.layout(), [self.start_btn])

        ui.ScrollArea.make_scrollable(
            container.layout, self, max_wh=[MAX_W, MAX_H]
        )
        self._set_io_visibility()
        container.setSizePolicy(
            QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding
        )

    def _start(self):
        self.results_path.mkdir(exist_ok=True, parents=True)

        if self.layer_choice:
            if self.label_layer_loader.layer_data() is not None:
                layer = self.label_layer_loader.layer()

                data = np.array(layer.data)
                instance = self.instance_widgets.run_method(data)

                save_layer(
                    self.results_path,
                    f"instance_{layer.name}_{utils.get_date_time()}.tif",
                    instance,
                )
                self._viewer.add_labels(
                    instance, name=f"instance_{layer.name}"
                )

        elif self.folder_choice.isChecked():
            if len(self.images_filepaths) != 0:
                images = [
                    self.instance_widgets.run_method(imread(file))
                    for file in self.images_filepaths
                ]
                save_folder(
                    self.results_path,
                    f"instance_results_{utils.get_date_time()}",
                    images,
                    self.images_filepaths,
                )


class ThresholdUtils(BasePluginFolder):
    """
    Creates a ThresholdUtils widget
    Args:
        viewer: viewer in which to process data
        parent: parent widget
    """

    def __init__(self, viewer: "napari.viewer.Viewer", parent=None):
        super().__init__(
            viewer,
            parent,
            loads_labels=False,
        )

        self.data_panel = self._build_io_panel()
        self._set_io_visibility()

        self.image_layer_loader.layer_list.label.setText("Layer :")
        self.image_layer_loader.set_layer_type(napari.layers.Layer)

        self.start_btn = ui.Button("Start", self._start)
        self.binarize_counter = ui.DoubleIncrementCounter(
            lower=0.0,
            upper=100000.0,
            step=0.5,
            default=10.0,
            text_label="Remove all smaller than (value):",
        )

        self.results_path = Path.home() / Path("cellseg3d/threshold")
        self.results_filewidget.text_field.setText(str(self.results_path))
        self.results_filewidget.check_ready()

        self.container = self._build()

        self.function = threshold

    def _build(self):
        container = ui.ContainerWidget()

        ui.add_widgets(
            self.data_panel.layout,
            [
                self.binarize_counter.label,
                self.binarize_counter,
                self.start_btn,
            ],
        )
        container.layout.addWidget(self.data_panel)

        ui.ScrollArea.make_scrollable(
            container.layout, self, max_wh=[MAX_W, MAX_H]
        )
        self._set_io_visibility()
        container.setSizePolicy(
            QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding
        )

        return container

    def _start(self):
        self.results_path.mkdir(exist_ok=True, parents=True)
        remove_size = self.binarize_counter.value()

        if self.layer_choice:
            if self.image_layer_loader.layer_data() is not None:
                layer = self.image_layer_loader.layer()

                data = np.array(layer.data)
                removed = self.function(data, remove_size)

                save_layer(
                    self.results_path,
                    f"threshold_{layer.name}_{utils.get_date_time()}.tif",
                    removed,
                )
                show_result(
                    self._viewer, layer, removed, f"threshold{layer.name}"
                )
        elif self.folder_choice.isChecked():
            if len(self.images_filepaths) != 0:
                images = [
                    self.function(imread(file), remove_size)
                    for file in self.images_filepaths
                ]
                save_folder(
                    self.results_path,
                    f"threshold_results_{utils.get_date_time()}",
                    images,
                    self.images_filepaths,
                )


# class ConvertUtils(BasePluginFolder):
#     """Utility widget that allows to convert labels from instance to semantic and the reverse."""
#
#     def __init__(self, viewer: "napari.viewer.Viewer", parent):
#         """Builds a ConvertUtils widget with the following buttons:
#
#         * A button to convert a folder of labels to semantic labels
#
#         * A button to convert a folder of labels to instance labels
#
#         * A button to convert a currently selected layer to semantic labels
#
#         * A button to convert a currently selected layer to instance labels
#         """
#
#         super().__init__(viewer, parent)
#         self._viewer = viewer
#         pass
