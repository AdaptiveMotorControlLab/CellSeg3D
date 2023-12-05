"""Several image processing utilities."""
from pathlib import Path
from warnings import warn

import napari
import numpy as np
import pandas as pd
from qtpy.QtWidgets import QLineEdit, QSizePolicy
from tifffile import imread

import napari_cellseg3d.interface as ui
from napari_cellseg3d import utils
from napari_cellseg3d.code_models.instance_segmentation import (
    InstanceWidgets,
    clear_large_objects,
    clear_small_objects,
    threshold,
    to_semantic,
    volume_stats,
)
from napari_cellseg3d.code_plugins.plugin_base import BasePluginUtils
from napari_cellseg3d.dev_scripts.crop_data import crop_3d_image

MAX_W = ui.UTILS_MAX_WIDTH
MAX_H = ui.UTILS_MAX_HEIGHT

logger = utils.LOGGER


class ArtifactRemovalUtils(BasePluginUtils):
    """Class to remove artifacts from images by removing large objects."""

    save_path = Path.home() / "cellseg3d" / "artifact_removed"

    def __init__(self, viewer: "napari.Viewer.viewer", parent=None):
        """Creates a ArtifactRemovalUtils widget.

        Args:
            viewer: viewer in which to process data
            parent: parent widget
        """
        super().__init__(
            viewer,
            parent,
            loads_labels=True,
            loads_images=False,
        )
        self.data_panel = self._build_io_panel()
        self.container = None
        self.start_btn = ui.Button("Start", self._start)

        self.artifact_size_counter = ui.IntIncrementCounter(
            lower=0,
            upper=100000,
            default=100,
            text_label="Remove all larger than\n(volume in pxs):",
        )

        self.label_layer_loader.layer_list.label.setText("Layer :")
        self.label_layer_loader.set_layer_type(napari.layers.Labels)

        self.results_path = str(self.save_path)
        self.results_filewidget.text_field.setText(str(self.results_path))
        self.results_filewidget.check_ready()

        self._build()

    def _build(self):
        container = ui.ContainerWidget()
        self.container = container

        ui.add_widgets(
            self.data_panel.layout,
            [
                self.data_panel,
                ui.add_blank(container),
                self.artifact_size_counter.label,
                self.artifact_size_counter,
                # ui.add_blank(container),
                self.start_btn,
            ],
        )
        container.layout.addWidget(self.data_panel)
        ui.ScrollArea.make_scrollable(
            container.layout,
            self,
            max_wh=[MAX_W, MAX_H],
        )
        self._set_io_visibility()
        container.setSizePolicy(
            QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding
        )
        return container

    def _remove_large(self, data, size):
        return clear_large_objects(data, size).astype(np.uint16)

    def _start(self):
        utils.mkdir_from_str(self.results_path)
        remove_size = self.artifact_size_counter.value()

        if self.layer_choice.isChecked():
            if self.label_layer_loader.layer_data() is not None:
                layer = self.label_layer_loader.layer()

                data = np.array(layer.data)
                removed = self._remove_large(data, remove_size)

                utils.save_layer(
                    self.results_path,
                    f"artifact_removed_{layer.name}_{utils.get_date_time()}.tif",
                    removed,
                )
                self.layer = utils.show_result(
                    self._viewer,
                    layer,
                    removed,
                    f"artifact_removed_{layer.name}",
                    existing_layer=self.layer,
                    add_as_labels=True,
                )
        elif (
            self.folder_choice.isChecked() and len(self.labels_filepaths) != 0
        ):
            images = [
                self._remove_large(imread(file), remove_size)
                for file in self.labels_filepaths
            ]
            utils.save_folder(
                self.results_path,
                f"artifact_removed_results_{utils.get_date_time()}",
                images,
                self.labels_filepaths,
            )
        else:
            logger.warning("Please specify a layer or a folder")


class FragmentUtils(BasePluginUtils):
    """Class to crop large 3D volumes into smaller fragments."""

    save_path = Path.home() / "cellseg3d" / "fragmented"

    def __init__(self, viewer: "napari.Viewer.viewer", parent=None):
        """Creates a FragmentUtils widget.

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
        self.start_btn = ui.Button("Start", self._start)
        self.size_selection = ui.AnisotropyWidgets(
            parent=self,
            default_x=64,
            default_y=64,
            default_z=64,
            always_visible=True,
            use_integer_counter=True,
        )
        [
            lbl.setText(f"Size in {ax} (pixels):")
            for lbl, ax in zip(self.size_selection.box_widgets_lbl, "xyz")
        ]
        [
            w.setToolTip(f"Size of crop for {dim} axis")
            for w, dim in zip(self.size_selection.box_widgets, "xyz")
        ]

        self.image_layer_loader.layer_list.label.setText("Layer :")
        self.image_layer_loader.set_layer_type(napari.layers.Layer)

        self.results_path = str(self.save_path)
        self.results_filewidget.text_field.setText(str(self.results_path))
        self.results_filewidget.check_ready()

        self._build()

    def _build(self):
        container = ui.ContainerWidget()

        ui.add_widgets(
            container.layout,
            [
                self.data_panel,
                self.size_selection,
                self.start_btn,
            ],
        )

        ui.ScrollArea.make_scrollable(
            container.layout,
            self,
            max_wh=[MAX_W, MAX_H],
        )

        self._set_io_visibility()
        self.setSizePolicy(
            QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding
        )

    def _fragment(self, crops, name):
        dir_name = f"/{name}_fragmented_{utils.get_time_filepath()}"
        Path(self.results_path + dir_name).mkdir(parents=True, exist_ok=False)
        for i, crop in enumerate(crops):
            utils.save_layer(
                self.results_path + dir_name,
                f"{name}_fragmented_{i}.tif",
                crop,
            )

    def _start(self):
        sizes = self.size_selection.resolution_zyx()
        if self.layer_choice.isChecked():
            layer = self.image_layer_loader.layer()
            crops = crop_3d_image(layer.data, sizes)
            self._fragment(crops, layer.name)
        elif self.folder_choice.isChecked():
            paths = self.images_filepaths
            for path in paths:
                data = imread(path)
                crops = crop_3d_image(data, sizes)
                self._fragment(crops, Path(path).stem)


class AnisoUtils(BasePluginUtils):
    """Class to correct anisotropy in images."""

    save_path = Path.home() / "cellseg3d" / "anisotropy"

    def __init__(self, viewer: "napari.Viewer.viewer", parent=None):
        """Creates a AnisoUtils widget.

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
        self.results_path = str(self.save_path)

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
                if isinstance(layer, napari.layers.Image):
                    isotropic_image = isotropic_image.astype(layer.data.dtype)
                else:
                    isotropic_image = isotropic_image.astype(np.uint16)
                self.layer = utils.show_result(
                    self._viewer,
                    layer,
                    isotropic_image,
                    f"isotropic_{layer.name}",
                    existing_layer=self.layer,
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


class RemoveSmallUtils(BasePluginUtils):
    """Widget to remove small objects."""

    save_path = Path.home() / "cellseg3d" / "small_removed"

    def __init__(self, viewer: "napari.viewer.Viewer", parent=None):
        """Creates a RemoveSmallUtils widget.

        Args:
            viewer: viewer in which to process data
            parent: parent widget
        """
        super().__init__(
            viewer,
            parent=parent,
            loads_images=False,
        )

        self.data_panel = self._build_io_panel()

        self.label_layer_loader.layer_list.label.setText("Layer :")

        self.start_btn = ui.Button("Start", self._start)
        self.size_for_removal_counter = ui.DoubleIncrementCounter(
            lower=0.0,
            upper=100000.0,
            default=10.0,
            text_label="Remove all smaller than (pxs):",
        )

        self.results_path = str(self.save_path)
        self.results_filewidget.text_field.setText(self.results_path)
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

        if self.layer_choice.isChecked():
            if self.label_layer_loader.layer_data() is not None:
                layer = self.label_layer_loader.layer()

                data = np.array(layer.data)
                removed = self.function(data, remove_size)

                utils.save_layer(
                    self.results_path,
                    f"cleared_{layer.name}_{utils.get_date_time()}.tif",
                    removed,
                )
                self.layer = utils.show_result(
                    self._viewer,
                    layer,
                    removed,
                    f"cleared_{layer.name}",
                    existing_layer=self.layer,
                    add_as_labels=True,
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


class ToSemanticUtils(BasePluginUtils):
    """Widget to create semantic labels from instance labels."""

    save_path = Path.home() / "cellseg3d" / "semantic_labels"

    def __init__(self, viewer: "napari.viewer.Viewer", parent=None):
        """Creates a ToSemanticUtils widget.

        Args:
            viewer: viewer in which to process data
            parent: parent widget
        """
        super().__init__(
            viewer,
            parent=parent,
            loads_images=False,
        )

        self.data_panel = self._build_io_panel()
        self.start_btn = ui.Button("Start", self._start)

        self.results_path = str(self.save_path)
        self.results_filewidget.text_field.setText(self.results_path)
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
        logger.debug(f"Running on layer : {self.layer_choice.isChecked()}")
        logger.debug(f"Running on folder : {self.folder_choice.isChecked()}")
        logger.debug(f"Images : {self.labels_filepaths}")

        self.results_filewidget.check_ready()

        if self.layer_choice.isChecked():
            if self.label_layer_loader.layer_data() is not None:
                layer = self.label_layer_loader.layer()

                data = np.array(layer.data)
                semantic = to_semantic(data)

                utils.save_layer(
                    self.results_path,
                    f"semantic_{layer.name}_{utils.get_date_time()}.tif",
                    semantic,
                )
                self.layer = utils.show_result(
                    self._viewer,
                    layer,
                    semantic,
                    f"semantic_{layer.name}",
                    existing_layer=self.layer,
                )
        elif (
            self.folder_choice.isChecked() and len(self.labels_filepaths) != 0
        ):
            images = [
                to_semantic(file, is_file_path=True)
                for file in self.labels_filepaths
            ]
            utils.save_folder(
                self.results_path,
                f"semantic_results_{utils.get_date_time()}",
                images,
                self.labels_filepaths,
            )
        else:
            logger.warning("Please specify a layer or a folder")


class ToInstanceUtils(BasePluginUtils):
    """Widget to convert semantic labels to instance labels."""

    save_path = Path.home() / "cellseg3d" / "instance_labels"

    def __init__(self, viewer: "napari.viewer.Viewer", parent=None):
        """Creates a ToInstanceUtils widget.

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
        self._set_io_visibility()

        self.image_layer_loader.layer_list.label.setText("Layer :")
        self.image_layer_loader.set_layer_type(napari.layers.Layer)

        self.instance_widgets = InstanceWidgets(parent=self)
        self.start_btn = ui.Button("Start", self._start)

        self.results_path = str(self.save_path)
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
        utils.mkdir_from_str(str(self.results_path))

        if self.layer_choice.isChecked():
            if self.image_layer_loader.layer_data() is not None:
                layer = self.image_layer_loader.layer()

                data = np.array(layer.data)
                instance = self.instance_widgets.run_method(data)

                utils.save_layer(
                    self.results_path,
                    f"instance_{layer.name}_{utils.get_date_time()}.tif",
                    instance,
                )
                self.layer = utils.show_result(
                    self._viewer,
                    layer,
                    instance,
                    f"instance_{layer.name}",
                    existing_layer=self.layer,
                    add_as_labels=True,
                )

        elif (
            self.folder_choice.isChecked() and len(self.images_filepaths) != 0
        ):
            images = [
                self.instance_widgets.run_method(imread(file))
                for file in self.images_filepaths
            ]
            utils.save_folder(
                self.results_path,
                f"instance_results_{utils.get_date_time()}",
                images,
                self.images_filepaths,
            )


class ThresholdUtils(BasePluginUtils):
    """Creates a ThresholdUtils widget.

    Args:
        viewer: viewer in which to process data
        parent: parent widget.
    """

    save_path = Path.home() / "cellseg3d" / "threshold"

    def __init__(self, viewer: "napari.viewer.Viewer", parent=None):
        """Creates a ThresholdUtils widget."""
        super().__init__(
            viewer,
            parent=parent,
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

        self.results_path = str(Path.home() / "cellseg3d" / "threshold")
        self.results_path = str(self.save_path)
        self.results_filewidget.text_field.setText(self.results_path)
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

        if self.layer_choice.isChecked():
            if self.image_layer_loader.layer_data() is not None:
                layer = self.image_layer_loader.layer()

                data = np.array(layer.data)
                removed = self.function(data, remove_size)

                utils.save_layer(
                    self.results_path,
                    f"threshold_{layer.name}_{utils.get_date_time()}.tif",
                    removed,
                )
                self.layer = utils.show_result(
                    self._viewer,
                    layer,
                    removed,
                    f"threshold_{layer.name}",
                    existing_layer=self.layer,
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


class StatsUtils(BasePluginUtils):
    """Widget to save statistics of a labels layer."""

    save_path = Path.home() / "cellseg3d" / "stats"

    def __init__(self, viewer: "napari.viewer.Viewer", parent=None):
        """Creates a StatsUtils widget.

        Args:
            viewer: viewer in which to process data
            parent: parent widget
        """
        super().__init__(
            viewer,
            parent=parent,
            loads_images=False,
        )

        self.data_panel = self._build_io_panel()

        self.csv_name_label = ui.make_label("CSV name :")
        self.csv_name = QLineEdit("volume_stats", parent=self)
        self.csv_name.setToolTip(
            "Name of the csv file.\nThe extension is added automatically;\nif running on a folder, the id of the image will be added to the name."
        )

        self.start_btn = ui.Button("Start", self._start)

        self.results_path = str(self.save_path)
        self.results_filewidget.text_field.setText(self.results_path)
        self.results_filewidget.check_ready()

        self.container = self._build()

    def _build(self):
        container = ui.ContainerWidget()

        ui.add_widgets(
            self.data_panel.layout,
            [
                self.csv_name_label,
                self.csv_name,
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
        if self.layer_choice.isChecked():
            if self.label_layer_loader.layer_data() is not None:
                layer = self.label_layer_loader.layer()

                data = np.array(layer.data)
                stats = volume_stats(data)
                if stats is None:
                    logger.warning(
                        "No stats to save, please ensure your label array is not empty"
                    )
                    return

                stats_df = pd.DataFrame(stats.get_dict())
                csv_name = self.csv_name.text() + ".csv"
                logger.info(f"Saving stats to {self.results_path}/{csv_name}")
                stats_df.to_csv(
                    self.results_path + "/" + csv_name, index=False
                )
        elif (
            self.folder_choice.isChecked() and len(self.labels_filepaths) != 0
        ):
            for i, image_path in enumerate(self.labels_filepaths):
                logger.debug(f"Loading image {i} : {image_path}")
                image = imread(image_path)
                image_name = Path(image_path).stem
                if image.sum() == 0:
                    m = f"Image {i} is empty, skipping."
                    logger.warning(m)
                    warn(m, stacklevel=0)
                    continue
                if not np.issubdtype(image.dtype, np.integer):
                    m = f"Image {i} is not integer, skipping. Make sure your labels are saved as integer values"
                    logger.warning(m)
                    warn(m, stacklevel=0)
                    continue
                logger.debug(f"Image {i} has shape {image.shape}")
                try:
                    stats = volume_stats(image)
                except Exception as e:
                    logger.warning(
                        f"Error computing stats for image {i} : {e}"
                    )
                    warn(f"Error computing stats for image {i}", stacklevel=0)
                logger.debug("Computing stats")
                stats_df = pd.DataFrame(stats.get_dict())
                csv_name = (
                    str(image_name) + "_" + self.csv_name.text() + f"_{i}.csv"
                )
                logger.info(
                    f"Saving stats to {(Path(self.results_path) / Path(csv_name)).resolve()}"
                )
                stats_df.to_csv(
                    Path(self.results_path) / Path(csv_name), index=False
                )
        else:
            logger.warning("Please specify a layer or a folder")
