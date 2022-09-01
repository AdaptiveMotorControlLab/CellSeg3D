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
from napari_cellseg3d.model_instance_seg import to_instance
from napari_cellseg3d.model_instance_seg import to_semantic
from napari_cellseg3d.plugin_base import BasePluginFolder

# TODO break down into multiple mini-widgets


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
        viewer.add_image(image, name=name)
    else:
        warnings.warn(
            f"Results not shown, unsupported layer type {type(layer)}"
        )


class AnisoUtils(BasePluginFolder):
    def __init__(self, viewer: "napari.Viewer.viewer", parent=None):

        super().__init__(
            viewer,
            parent,
            loads_images=True,
            loads_labels=False,
            has_results=True,
        )

        self.data_panel = self.build_io_panel()

        self.image_layer_loader.layer_list.label.setText("Layer :")
        self.image_layer_loader.layer_type = napari.layers.Layer

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
            container.layout, self, min_wh=[100, 200], base_wh=[100, 200]
        )

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
            loads_images=True,
            loads_labels=False,
            has_results=True,
        )

        self.data_panel = self.build_io_panel()

        self.image_layer_loader.layer_list.label.setText("Layer :")
        self.image_layer_loader.layer_type = napari.layers.Layer

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

        self._build()

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

        ui.ScrollArea.make_scrollable(container.layout, self)

    def _start(self):
        self.results_path.mkdir(exist_ok=True)
        remove_size = self.size_for_removal_counter.value()

        if self.layer_choice:
            if self.image_layer_loader.layer_data() is not None:
                layer = self.image_layer_loader.layer()

                data = np.array(layer.data, dtype=np.int16)
                removed = clear_small_objects(data, remove_size)

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
            loads_labels=True,
            has_results=True,
        )

        self.data_panel = self.build_io_panel()

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

        ui.ScrollArea.make_scrollable(container.layout, self)

    def _start(self):
        self.results_path.mkdir(exist_ok=True)

        if self.layer_choice:
            if self.image_layer_loader.layer_data() is not None:
                layer = self.image_layer_loader.layer()

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

    def __init__(self, parent = None):

        super().__init__(parent)

        self.method_choice = ui.DropdownMenu(
            config.INSTANCE_SEGMENTATION_METHOD_LIST.keys()
        )
        self.method = config.INSTANCE_SEGMENTATION_METHOD_LIST[self.method_choice.currentText()]

        self.threshold_slider1 = ui.Slider(
            lower=1,
            upper=99,
            default=config.PostProcessConfig().instance.threshold.threshold_value
            * 100,
            divide_factor=100.0,
            step=5,
            text_label="Probability threshold :",
        )
        self.threshold_slider2 = ui.Slider(
            lower=1,
            upper=99,
            default=config.PostProcessConfig().instance.threshold.threshold_value
                    * 100,
            divide_factor=100.0,
            step=5,
            text_label="Probability threshold :",
        )

        self.counter1 = ui.IntIncrementCounter(
            upper=100,
            default=30,
            step=5,
            label="Small object removal (pxs) :",
        )
        self.counter2 = ui.IntIncrementCounter(
            upper=100,
            default=10,
            step=5,
            label="Small seed removal (pxs) :",
        )

    def _show_watershed(self):
        name = config.INSTANCE_SEGMENTATION_METHOD_LIST.keys()[0]
        if self.method_choice.currentText() == name:

            self.threshold_slider1.container.setVisible(True)
            self.threshold_slider2.container.setVisible(True)
            self.counter1.setVisible(True)
            self.counter2.setVisible(True)

            self.method = config.INSTANCE_SEGMENTATION_METHOD_LIST[name]

    def _show_connected(self):
        name = config.INSTANCE_SEGMENTATION_METHOD_LIST.keys()[1]
        if self.method_choice.currentText() == name:

            self.threshold_slider1.container.setVisible(True)
            self.threshold_slider2.container.setVisible(False)
            self.counter1.setVisible(True)
            self.counter2.setVisible(False)

            self.method = config.INSTANCE_SEGMENTATION_METHOD_LIST[name]

    def _show_slider1(self, is_visible:bool = True):
        self.threshold_slider1.container.setVisible(is_visible)

    def _show_slider2(self, is_visible:bool = True):
        self.threshold_slider2.container.setVisible(is_visible)

    def _show_counter1(self, is_visible:bool = True):
        self.counter1.setVisible(is_visible)
        self.counter1.label.setVisible(is_visible)


class ToInstanceUtils(BasePluginFolder):

    def __init__(self, viewer: "napari.viewer.Viewer", parent=None):

        super().__init__(
            viewer,
            parent,
            loads_images=False,
            loads_labels=True,
            has_results=True,
        )

        self.data_panel = self.build_io_panel()

        self.start_btn = ui.Button("Start", self._start)

        self.results_path = Path.home() / Path("cellseg3d/instance")
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

        ui.ScrollArea.make_scrollable(container.layout, self)

    def _start(self):
        self.results_path.mkdir(exist_ok=True)

        if self.layer_choice:
            if self.image_layer_loader.layer_data() is not None:
                layer = self.image_layer_loader.layer()

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


class ThresholdUtils(BasePluginFolder):
    pass


class ConvertUtils(BasePluginFolder):
    """Utility widget that allows to convert labels from instance to semantic and the reverse."""

    def __init__(self, viewer: "napari.viewer.Viewer", parent):
        """Builds a ConvertUtils widget with the following buttons:

        * A button to convert a folder of labels to semantic labels

        * A button to convert a folder of labels to instance labels

        * A button to convert a currently selected layer to semantic labels

        * A button to convert a currently selected layer to instance labels
        """

        super().__init__(viewer, parent)

        self._viewer = viewer

        # FIXME test
        viewer.window.add_dock_widget(AnisoUtils(viewer, self))


#
#         ########################
#         # interface
#
#         # label conversion
#         self.btn_convert_folder_semantic = ui.Button(
#             "Convert to semantic labels", func=self.folder_to_semantic
#         )
#         self.btn_convert_layer_semantic = ui.Button(
#             "Convert to semantic labels", func=self.layer_to_semantic
#         )
#         self.btn_convert_folder_instance = ui.Button(
#             "Convert to instance labels", func=self.folder_to_instance
#         )
#         self.btn_convert_layer_instance = ui.Button(
#             "Convert to instance labels", func=self.layer_to_instance
#         )
#         # remove small
#         self.btn_remove_small_folder = ui.Button(
#             "Remove small in folder", func=self.folder_remove_small
#         )
#         self.btn_remove_small_layer = ui.Button(
#             "Remove small in layer", func=self.layer_remove_small
#         )
#         self.small_object_thresh_choice = ui.IntIncrementCounter(
#             lower=1, upper=1000, default=15
#         )
#
#         # convert anisotropy
#         self.anisotropy_converter = ui.AnisotropyWidgets(
#             parent=self, always_visible=True
#         )
#         self.btn_aniso_folder = ui.Button(
#             "Correct anisotropy in folder", self.folder_anisotropy, self
#         )
#         self.btn_aniso_layer = ui.Button(
#             "Correct anisotropy in layer", self.layer_anisotropy, self
#         )
#
#         self.lbl_error = ui.make_label("", self)
#         self.lbl_error.setVisible(False)
#
#         self.image_filewidget.button.setVisible(False)
#         self.image_filewidget.text_field.setVisible(False)
#
#         # self.results_filewidget.set_required(True)
#         self.labels_filewidget.required = False
#         # TODO improve not ready check for labels since optional until using folder conversion
#         ###############################
#         # tooltips
#         self.btn_convert_folder_semantic.setToolTip(
#             "Convert specified folder to semantic (0/1)"
#         )
#         self.btn_convert_folder_instance.setToolTip(
#             "Convert specified folder to instance (unique ID per object)"
#         )
#         self.btn_convert_layer_instance.setToolTip(
#             "Convert currently selected layer to instance (unique ID per object)"
#         )
#         self.btn_convert_layer_semantic.setToolTip(
#             "Convert currently selected layer to semantic (0/1)"
#         )
#
#         self.btn_remove_small_layer.setToolTip(
#             "Remove small objects on selected layer image"
#         )
#         self.btn_remove_small_folder.setToolTip(
#             "Remove small objects in all images of selected folder"
#         )
#         self.small_object_thresh_choice.setToolTip(
#             "All objects in the image smaller in volume than this number of pixels will be removed"
#         )
#         self.btn_aniso_layer.setToolTip(
#             "Resize the selected layer to be isotropic, based on the chosen resolutions above."
#             "\nDOES NOT WORK WITH INSTANCE LABELS, CONVERT TO SEMANTIC FIRST"
#         )
#         self.btn_aniso_folder.setToolTip(
#             "Resize the images in the selected folder to be isotropic, based on the chosen resolutions above."
#             "\nDOES NOT WORK WITH INSTANCE LABELS, CONVERT TO SEMANTIC FIRST"
#         )
#         ###############################
#
#         self.build()
#
#     def build(self):
#         """Builds the layout of the widget with the following buttons :
#
#         * Set path to results
#
#         * Set path to labels
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
#         l, t, r, b = 7, 20, 7, 11
#
#         w = ui.ContainerWidget()
#         layout = w.layout
#
#         results_widget = ui.combine_blocks(
#             right_or_below=self.results_filewidget.button,
#             left_or_above=self.results_filewidget.text_field,
#             min_spacing=70,
#         )
#
#         ui.GroupedWidget.create_single_widget_group(
#             "Results",
#             results_widget,
#             layout,
#             l=3,
#             t=11,
#             r=3,
#             b=3,
#         )
#         ###############################
#         ui.add_blank(layout=layout, widget=self)
#         ###############################
#         aniso_group_w, aniso_group_l = ui.make_group(
#             "Correct anisotropy", l, t, r, b, parent=None
#         )
#
#         ui.add_widgets(
#             aniso_group_l,
#             [
#                 self.anisotropy_converter,
#             ],
#             ui.LEFT_AL,
#         )
#
#         aniso_group_w.setLayout(aniso_group_l)
#         layout.addWidget(aniso_group_w)
#
#         ###############################
#         ui.add_blank(layout=layout, widget=self)
#         #############################################################
#         small_group_w, small_group_l = ui.make_group(
#             "Remove small objects", l, t, r, b, parent=None
#         )
#
#         ui.add_widgets(
#             small_group_l,
#             [
#                 self.small_object_thresh_choice,
#             ],
#             ui.HCENTER_AL,
#         )
#
#         small_group_w.setLayout(small_group_l)
#         layout.addWidget(small_group_w)
#         #########################################
#         ui.add_blank(layout=layout, widget=self)
#         #############################################################
#         layer_group_w, layer_group_l = ui.make_group(
#             "Convert selected layer", l, t, r, b, parent=None
#         )
#
#         ui.add_widgets(
#             layer_group_l,
#             [
#                 self.btn_convert_layer_instance,
#                 self.btn_convert_layer_semantic,
#                 self.btn_remove_small_layer,
#                 self.btn_aniso_layer,
#             ],
#             ui.HCENTER_AL,
#         )
#
#         layer_group_w.setLayout(layer_group_l)
#         layout.addWidget(layer_group_w)
#         ###############################
#         ui.add_blank(layout=layout, widget=self)
#         ###############################
#         folder_group_w, folder_group_l = ui.make_group(
#             "Convert folder", l, t, r, b, parent=None
#         )
#
#         folder_group_l.addWidget(
#             ui.combine_blocks(
#                 right_or_below=self.labels_filewidget.button,
#                 left_or_above=self.labels_filewidget.text_field,
#                 min_spacing=70,
#             )
#         )
#
#         ui.add_widgets(
#             folder_group_l,
#             [
#                 self.btn_convert_folder_instance,
#                 self.btn_convert_folder_semantic,
#                 self.btn_remove_small_folder,
#                 self.btn_aniso_folder,
#             ],
#             ui.HCENTER_AL,
#         )
#
#         folder_group_w.setLayout(folder_group_l)
#         layout.addWidget(folder_group_w)
#         ###############################
#         ui.add_blank(layout=layout, widget=self)
#
#         ui.add_widgets(
#             layout,
#             [
#                 ui.add_blank(self),
#                 self.make_close_button(),
#                 ui.add_blank(self),
#                 self.lbl_error,
#             ],
#         )
#
#         ui.ScrollArea.make_scrollable(
#             layout, self, min_wh=[230, 400], base_wh=[230, 450]
#         )
#
#     def folder_to_semantic(self):
#         """Converts folder of labels to semantic labels"""
#         if not self.check_ready_folder():
#             return
#
#         folder_name = f"converted_to_semantic_labels_{utils.get_date_time()}"
#
#         images = [
#             to_semantic(file, is_file_path=True)
#             for file in self.labels_filepaths
#         ]
#
#         self.save_folder(folder_name, images)
#
#     def layer_to_semantic(self):
#         """Converts selected layer to semantic labels"""
#         if not self.check_ready_layer():
#             return
#
#         im = self._viewer.layers.selection.active.data
#         name = self._viewer.layers.selection.active.name
#         semantic_labels = to_semantic(im)
#
#         self.save_layer(
#             f"{name}_semantic_{utils.get_time_filepath()}"
#             + self.filetype_choice.currentText(),
#             semantic_labels,
#         )
#
#         self._viewer.add_labels(semantic_labels, name=f"converted_semantic")
#
#     def folder_to_instance(self):
#         """Converts the chosen folder to instance labels"""
#         if not self.check_ready_folder():
#             return
#
#         images = [
#             to_instance(file, is_file_path=True)
#             for file in self.labels_filepaths
#         ]
#
#         self.save_folder(
#             f"converted_to_instance_labels_{utils.get_date_time()}", images
#         )
#
#     def layer_to_instance(self):
#         """Converts the selected layer to instance labels"""
#         if not self.check_ready_layer():
#             return
#
#         im = [self._viewer.layers.selection.active.data]
#         name = self._viewer.layers.selection.active.name
#         instance_labels = to_instance(im)
#
#         self.save_layer(
#             f"{name}_instance_{utils.get_time_filepath()}"
#             + self.filetype_choice.currentText(),
#             instance_labels,
#         )
#
#         self._viewer.add_labels(instance_labels, name=f"converted_instance")
#
#     def layer_remove_small(self):
#         """Removes small objects in selected layer"""
#         if not self.check_ready_layer():
#             return
#
#         im = self._viewer.layers.selection.active.data
#         name = self._viewer.layers.selection.active.name
#
#         cleared_labels = clear_small_objects(
#             im, self.small_object_thresh_choice.value()
#         )
#
#         self.save_layer(
#             f"{name}_cleared_{utils.get_time_filepath()}"
#             + self.filetype_choice.currentText(),
#             cleared_labels,
#         )
#
#         self._viewer.add_image(cleared_labels, name=f"small_cleared")
#
#     def folder_remove_small(self):
#         """Removes small objects in folder of labels"""
#         if not self.check_ready_folder():
#             return
#
#         images = [
#             clear_small_objects(
#                 file,
#                 self.small_object_thresh_choice.value(),
#                 is_file_path=True,
#             )
#             for file in self.labels_filepaths
#         ]
#
#         self.save_folder(f"small_cleared_{utils.get_date_time()}", images)
#
#
#
#     def check_ready_folder(self):  # TODO add color change
#         """Check if results and source folders are correctly set"""
#         if self.results_path is None:
#             err = "ERROR : please set results folder"
#             print(err)
#             self.lbl_error.setText(err)
#             self.lbl_error.setVisible(True)
#             return False
#         if self.labels_filepaths != []:
#             self.lbl_error.setVisible(False)
#             return True
#
#         err = "ERROR : please set valid source labels folder"
#         print(err)
#         self.lbl_error.setText(err)
#         self.lbl_error.setVisible(True)
#         return False
#
#     def check_ready_layer(self):  # TODO add color change
#         """Check if results and layer are selected"""
#         if self.results_path is None:
#             err = "ERROR : please set results folder"
#             print(err)
#             self.lbl_error.setText(err)
#             self.lbl_error.setVisible(True)
#             return False
#         if self._viewer.layers.selection.active is None:
#             err = "ERROR : Please select a single layer"
#             print(err)
#             self.lbl_error.setText(err)
#             self.lbl_error.setVisible(True)
#             return False
#         self.lbl_error.setVisible(False)
#         return True
#
#     def save_layer(self, file_name, image):
#
#         path = os.path.join(self.results_path, file_name)
#         print(self.results_path)
#         print(path)
#
#         if self.results_path is not None:
#             imwrite(
#                 path,
#                 image,
#             )
#
#     def save_folder(self, folder_name, images):
#
#         results_folder = os.path.join(
#             self.results_path,
#             folder_name,
#         )
#
#         os.makedirs(results_folder, exist_ok=False)
#
#         for file, image in zip(self.labels_filepaths, images):
#
#             path = os.path.join(results_folder, os.path.basename(file))
#
#             imwrite(
#                 path,
#                 image,
#             )
