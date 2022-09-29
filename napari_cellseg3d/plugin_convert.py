from pathlib import Path
import warnings

import napari
import numpy as np
from tifffile import imread
from tifffile import imwrite

from qtpy.QtWidgets import QWidget

from napari_cellseg3d import config
import napari_cellseg3d.interface as ui
from napari_cellseg3d import utils
from napari_cellseg3d.model_instance_seg import clear_small_objects
from napari_cellseg3d.model_instance_seg import threshold
from napari_cellseg3d.model_instance_seg import to_semantic
from napari_cellseg3d.plugin_base import BasePluginFolder

# TODO break down into multiple mini-widgets
# TODO create parent class for utils modules to avoid duplicates

MAX_W = 200
MAX_H = 400


def save_folder(results_path, folder_name, images, image_paths):

    results_folder = results_path / Path(folder_name)
    results_folder.mkdir(exist_ok=False)

    for file, image in zip(image_paths, images):
        path = results_folder / Path(file).name

        imwrite(
            path,
            image,
        )
    print(f"Saved processed folder as : {results_folder}")


def save_layer(results_path, image_name, image):
    path = str(results_path / Path(image_name))  # TODO flexible filetype
    print(f"Saved as : {path}")
    imwrite(path, image)


def show_result(viewer, layer, image, name):

    if isinstance(layer, napari.layers.Image):
        viewer.add_image(image, name=name)
    elif isinstance(layer, napari.layers.Labels):
        viewer.add_labels(image, name=name)
    else:
        warnings.warn(
            f"Results not shown, unsupported layer type {type(layer)}"
        )


class AnisoUtils(BasePluginFolder):
    def __init__(self, viewer: "napari.Viewer.viewer", parent=None):

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

        self.set_io_visibility()

    def _start(self):

        self.results_path.mkdir(exist_ok=True)
        zoom = self.aniso_widgets.scaling_zyx()

        self._viewer.window.remove_dock_widget(self.parent())

        if self.layer_choice.isChecked():
            if self.image_layer_loader.layer_data() is not None:
                layer = self.image_layer_loader.layer()

                data = np.array(layer.data, dtype=np.int16)
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
                    utils.resize(np.array(imread(file), dtype=np.int16), zoom)
                    for file in self.images_filepaths
                ]
                save_folder(
                    self.results_path,
                    f"isotropic_results_{utils.get_date_time()}",
                    images,
                    self.images_filepaths,
                )


class RemoveSmallUtils(BasePluginFolder):
    def __init__(self, viewer: "napari.viewer.Viewer", parent=None):

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
            label="Remove all smaller than (pxs):",
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
        self.set_io_visibility()

        return container

    def _start(self):
        self.results_path.mkdir(exist_ok=True)
        remove_size = self.size_for_removal_counter.value()

        if self.layer_choice:
            if self.image_layer_loader.layer_data() is not None:
                layer = self.image_layer_loader.layer()

                data = np.array(layer.data, dtype=np.int16)
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


class ToSemanticUtils(BasePluginFolder):
    def __init__(self, viewer: "napari.viewer.Viewer", parent=None):

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
        self.set_io_visibility()

    def _start(self):
        self.results_path.mkdir(exist_ok=True)

        if self.layer_choice:
            if self.label_layer_loader.layer_data() is not None:
                layer = self.label_layer_loader.layer()

                data = np.array(layer.data, dtype=np.int16)
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


class InstanceWidgets(QWidget):
    def __init__(self, parent=None):

        super().__init__(parent)

        self.method_choice = ui.DropdownMenu(
            config.INSTANCE_SEGMENTATION_METHOD_LIST.keys()
        )
        self._method = config.INSTANCE_SEGMENTATION_METHOD_LIST[
            self.method_choice.currentText()
        ]

        self.method_choice.currentTextChanged.connect(self._show_connected)
        self.method_choice.currentTextChanged.connect(self._show_watershed)

        self.threshold_slider1 = ui.Slider(
            lower=0,
            upper=100,
            default=50,
            divide_factor=100.0,
            step=5,
            text_label="Probability threshold :",
        )
        """Base prob. threshold"""
        self.threshold_slider2 = ui.Slider(
            lower=0,
            upper=100,
            default=90,
            divide_factor=100.0,
            step=5,
            text_label="Probability threshold (seeding) :",
        )
        """Second prob. thresh. (seeding)"""

        self.counter1 = ui.IntIncrementCounter(
            upper=100,
            default=10,
            step=5,
            label="Small object removal (pxs) :",
        )
        """Small obj. rem."""

        self.counter2 = ui.IntIncrementCounter(
            upper=100,
            default=3,
            step=5,
            label="Small seed removal (pxs) :",
        )
        """Small seed rem."""

        self._build()

    def get_method(self, volume):
        return self._method(
            volume,
            self.threshold_slider1.slider_value,
            self.counter1.value(),
            self.threshold_slider2.slider_value,
            self.counter2.value(),
        )

    def _build(self):

        group = ui.GroupedWidget("Instance segmentation")

        ui.add_widgets(
            group.layout,
            [
                self.method_choice,
                self.threshold_slider1.container,
                self.threshold_slider2.container,
                self.counter1.label,
                self.counter1,
                self.counter2.label,
                self.counter2,
            ],
        )

        self.setLayout(group.layout)
        self._set_tooltips()

    def _set_tooltips(self):

        self.method_choice.setToolTip(
            "Choose which method to use for instance segmentation"
            "\nConnected components : all separated objects will be assigned an unique ID. "
            "Robust but will not work correctly with adjacent/touching objects\n"
            "Watershed : assigns objects ID based on the probability gradient surrounding an object. "
            "Requires the model to surround objects in a gradient;"
            " can possibly correctly separate unique but touching/adjacent objects."
        )
        self.threshold_slider1.tooltips = (
            "All objects below this probability will be ignored (set to 0)"
        )
        self.counter1.setToolTip(
            "Will remove all objects smaller (in volume) than the specified number of pixels"
        )
        self.threshold_slider2.tooltips = (
            "All seeds below this probability will be ignored (set to 0)"
        )
        self.counter2.setToolTip(
            "Will remove all seeds smaller (in volume) than the specified number of pixels"
        )

    def _show_watershed(self):
        name = "Watershed"
        if self.method_choice.currentText() == name:

            self._show_slider1()
            self._show_slider2()
            self._show_counter1()
            self._show_counter2()

            self._method = config.INSTANCE_SEGMENTATION_METHOD_LIST[name]

    def _show_connected(self):
        name = "Connected components"
        if self.method_choice.currentText() == name:

            self._show_slider1()
            self._show_slider2(False)
            self._show_counter1()
            self._show_counter2(False)

            self._method = config.INSTANCE_SEGMENTATION_METHOD_LIST[name]

    def _show_slider1(self, is_visible: bool = True):
        self.threshold_slider1.container.setVisible(is_visible)

    def _show_slider2(self, is_visible: bool = True):
        self.threshold_slider2.container.setVisible(is_visible)

    def _show_counter1(self, is_visible: bool = True):
        self.counter1.setVisible(is_visible)
        self.counter1.label.setVisible(is_visible)

    def _show_counter2(self, is_visible: bool = True):
        self.counter2.setVisible(is_visible)
        self.counter2.label.setVisible(is_visible)


class ToInstanceUtils(BasePluginFolder):
    def __init__(self, viewer: "napari.viewer.Viewer", parent=None):

        super().__init__(
            viewer,
            parent,
            loads_images=False,
        )

        self.data_panel = self._build_io_panel()
        self.label_layer_loader.set_layer_type(napari.layers.Layer)

        self.instance_widgets = InstanceWidgets()

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
        self.set_io_visibility()

    def _start(self):
        self.results_path.mkdir(exist_ok=True)

        if self.layer_choice:
            if self.label_layer_loader.layer_data() is not None:
                layer = self.label_layer_loader.layer()

                data = np.array(layer.data, dtype=np.int16)
                instance = self.instance_widgets.get_method(data)

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
                    self.instance_widgets.get_method(imread(file))
                    for file in self.images_filepaths
                ]
                save_folder(
                    self.results_path,
                    f"instance_results_{utils.get_date_time()}",
                    images,
                    self.images_filepaths,
                )


class ThresholdUtils(BasePluginFolder):
    def __init__(self, viewer: "napari.viewer.Viewer", parent=None):

        super().__init__(
            viewer,
            parent,
            loads_labels=False,
        )

        self.data_panel = self._build_io_panel()
        self.set_io_visibility()

        self.image_layer_loader.layer_list.label.setText("Layer :")
        self.image_layer_loader.set_layer_type(napari.layers.Layer)

        self.start_btn = ui.Button("Start", self._start)
        self.binarize_counter = ui.DoubleIncrementCounter(
            lower=0.0,
            upper=100000.0,
            step=0.5,
            default=10.0,
            label="Remove all smaller than (value):",
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
        self.set_io_visibility()

        return container

    def _start(self):
        self.results_path.mkdir(exist_ok=True)
        remove_size = self.binarize_counter.value()

        if self.layer_choice:
            if self.image_layer_loader.layer_data() is not None:
                layer = self.image_layer_loader.layer()

                data = np.array(layer.data, dtype=np.int16)
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
