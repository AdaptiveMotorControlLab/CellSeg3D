from pathlib import Path

import napari
import numpy as np
from qtpy.QtWidgets import QSizePolicy
from tifffile import imread

import napari_cellseg3d.interface as ui
from napari_cellseg3d import utils
from napari_cellseg3d.code_models.instance_segmentation import (
    InstanceWidgets,
    clear_small_objects,
    threshold,
    to_semantic,
)
from napari_cellseg3d.code_plugins.plugin_base import BasePluginFolder

MAX_W = ui.UTILS_MAX_WIDTH
MAX_H = ui.UTILS_MAX_HEIGHT

logger = utils.LOGGER


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

        self.results_path = str(Path.home() / Path("cellseg3d/anisotropy"))
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
        utils.mkdir_from_str(self.results_path)
        zoom = self.aniso_widgets.scaling_zyx()

        if self.layer_choice.isChecked():
            if self.image_layer_loader.layer_data() is not None:
                layer = self.image_layer_loader.layer()

                data = np.array(layer.data)
                isotropic_image = utils.resize(data, zoom)

                utils.save_layer(
                    self.results_path,
                    f"isotropic_{layer.name}_{utils.get_date_time()}.tif",
                    isotropic_image,
                )
                utils.show_result(
                    self._viewer,
                    layer,
                    isotropic_image,
                    f"isotropic_{layer.name}",
                )

        elif (
            self.folder_choice.isChecked() and len(self.images_filepaths) != 0
        ):
            images = [
                utils.resize(np.array(imread(file)), zoom)
                for file in self.images_filepaths
            ]
            utils.save_folder(
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
        utils.mkdir_from_str(self.results_path)
        remove_size = self.size_for_removal_counter.value()

        if self.layer_choice:
            if self.image_layer_loader.layer_data() is not None:
                layer = self.image_layer_loader.layer()

                data = np.array(layer.data)
                removed = self.function(data, remove_size)

                utils.save_layer(
                    self.results_path,
                    f"cleared_{layer.name}_{utils.get_date_time()}.tif",
                    removed,
                )
                utils.show_result(
                    self._viewer, layer, removed, f"cleared_{layer.name}"
                )
        elif (
            self.folder_choice.isChecked() and len(self.images_filepaths) != 0
        ):
            images = [
                clear_small_objects(file, remove_size, is_file_path=True)
                for file in self.images_filepaths
            ]
            utils.save_folder(
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

                utils.save_layer(
                    self.results_path,
                    f"semantic_{layer.name}_{utils.get_date_time()}.tif",
                    semantic,
                )
                utils.show_result(
                    self._viewer, layer, semantic, f"semantic_{layer.name}"
                )
        elif (
            self.folder_choice.isChecked() and len(self.images_filepaths) != 0
        ):
            images = [
                to_semantic(file, is_file_path=True)
                for file in self.images_filepaths
            ]
            utils.save_folder(
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
        utils.mkdir_from_str(self.results_path)

        if self.layer_choice:
            if self.label_layer_loader.layer_data() is not None:
                layer = self.label_layer_loader.layer()

                data = np.array(layer.data)
                instance = self.instance_widgets.run_method(data)

                utils.save_layer(
                    self.results_path,
                    f"instance_{layer.name}_{utils.get_date_time()}.tif",
                    instance,
                )
                self._viewer.add_labels(
                    instance, name=f"instance_{layer.name}"
                )

        elif (
            self.folder_choice.isChecked() and len(self.images_filepaths) != 0
        ):
            images = [
                self.instance_widgets.run_method_on_channels(imread(file))
                for file in self.images_filepaths
            ]
            utils.save_folder(
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
        utils.mkdir_from_str(self.results_path)
        remove_size = self.binarize_counter.value()

        if self.layer_choice:
            if self.image_layer_loader.layer_data() is not None:
                layer = self.image_layer_loader.layer()

                data = np.array(layer.data)
                removed = self.function(data, remove_size)

                utils.save_layer(
                    self.results_path,
                    f"threshold_{layer.name}_{utils.get_date_time()}.tif",
                    removed,
                )
                utils.show_result(
                    self._viewer, layer, removed, f"threshold{layer.name}"
                )
        elif (
            self.folder_choice.isChecked() and len(self.images_filepaths) != 0
        ):
            images = [
                self.function(imread(file), remove_size)
                for file in self.images_filepaths
            ]
            utils.save_folder(
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
