import os

import napari
import numpy as np
from tifffile import imwrite, imread

import napari_cellseg3d.interface as ui
from napari_cellseg3d import utils
from napari_cellseg3d.model_instance_seg import clear_small_objects
from napari_cellseg3d.model_instance_seg import to_instance
from napari_cellseg3d.model_instance_seg import to_semantic
from napari_cellseg3d.plugin_base import BasePluginFolder


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

        ########################
        # interface

        # label conversion
        self.btn_convert_folder_semantic = ui.Button(
            "Convert to semantic labels", func=self.folder_to_semantic
        )
        self.btn_convert_layer_semantic = ui.Button(
            "Convert to semantic labels", func=self.layer_to_semantic
        )
        self.btn_convert_folder_instance = ui.Button(
            "Convert to instance labels", func=self.folder_to_instance
        )
        self.btn_convert_layer_instance = ui.Button(
            "Convert to instance labels", func=self.layer_to_instance
        )
        # remove small
        self.btn_remove_small_folder = ui.Button(
            "Remove small in folder", func=self.folder_remove_small
        )
        self.btn_remove_small_layer = ui.Button(
            "Remove small in layer", func=self.layer_remove_small
        )
        self.small_object_thresh_choice = ui.IntIncrementCounter(
            min=1, max=1000, default=15
        )

        # convert anisotropy
        self.anisotropy_converter = ui.AnisotropyWidgets(
            parent=self, always_visible=True
        )
        self.btn_aniso_folder = ui.Button(
            "Correct anisotropy in folder", self.folder_anisotropy, self
        )
        self.btn_aniso_layer = ui.Button(
            "Correct anisotropy in layer", self.layer_anisotropy, self
        )

        self.lbl_error = ui.make_label("", self)
        self.lbl_error.setVisible(False)

        self.btn_image_files.setVisible(False)
        self.lbl_image_files.setVisible(False)

        # self.results_filewidget.set_required(True)
        self.label_filewidget.set_required(False)
        # TODO improve not ready check for labels since optional until using folder conversion
        ###############################
        # tooltips
        self.btn_convert_folder_semantic.setToolTip(
            "Convert specified folder to semantic (0/1)"
        )
        self.btn_convert_folder_instance.setToolTip(
            "Convert specified folder to instance (unique ID per object)"
        )
        self.btn_convert_layer_instance.setToolTip(
            "Convert currently selected layer to instance (unique ID per object)"
        )
        self.btn_convert_layer_semantic.setToolTip(
            "Convert currently selected layer to semantic (0/1)"
        )

        self.btn_remove_small_layer.setToolTip(
            "Remove small objects on selected layer image"
        )
        self.btn_remove_small_folder.setToolTip(
            "Remove small objects in all images of selected folder"
        )
        self.small_object_thresh_choice.setToolTip(
            "All objects in the image smaller in volume than this number of pixels will be removed"
        )
        self.btn_aniso_layer.setToolTip(
            "Resize the selected layer to be isotropic, based on the chosen resolutions above."
            "\nDOES NOT WORK WITH INSTANCE LABELS, CONVERT TO SEMANTIC FIRST"
        )
        self.btn_aniso_folder.setToolTip(
            "Resize the images in the selected folder to be isotropic, based on the chosen resolutions above."
            "\nDOES NOT WORK WITH INSTANCE LABELS, CONVERT TO SEMANTIC FIRST"
        )
        ###############################

        self.build()

    def build(self):
        """Builds the layout of the widget with the following buttons :

        * Set path to results

        * Set path to labels

        * A button to convert a folder of labels to semantic labels

        * A button to convert a folder of labels to instance labels

        * A button to convert a currently selected layer to semantic labels

        * A button to convert a currently selected layer to instance labels
        """

        l, t, r, b = 7, 20, 7, 11

        w, layout = ui.make_container()

        results_widget = ui.combine_blocks(
            right_or_below=self.btn_result_path,
            left_or_above=self.lbl_result_path,
            min_spacing=70,
        )

        ui.add_to_group(
            "Results",
            results_widget,
            layout,
            L=3,
            T=11,
            R=3,
            B=3,
        )
        ###############################
        ui.add_blank(layout=layout, widget=self)
        ###############################
        aniso_group_w, aniso_group_l = ui.make_group(
            "Correct anisotropy", l, t, r, b, parent=None
        )

        ui.add_widgets(
            aniso_group_l,
            [
                self.anisotropy_converter,
            ],
            ui.LEFT_AL,
        )

        aniso_group_w.setLayout(aniso_group_l)
        layout.addWidget(aniso_group_w)

        ###############################
        ui.add_blank(layout=layout, widget=self)
        #############################################################
        small_group_w, small_group_l = ui.make_group(
            "Remove small objects", l, t, r, b, parent=None
        )

        ui.add_widgets(
            small_group_l,
            [
                self.small_object_thresh_choice,
            ],
            ui.HCENTER_AL,
        )

        small_group_w.setLayout(small_group_l)
        layout.addWidget(small_group_w)
        #########################################
        ui.add_blank(layout=layout, widget=self)
        #############################################################
        layer_group_w, layer_group_l = ui.make_group(
            "Convert selected layer", l, t, r, b, parent=None
        )

        ui.add_widgets(
            layer_group_l,
            [
                self.btn_convert_layer_instance,
                self.btn_convert_layer_semantic,
                self.btn_remove_small_layer,
                self.btn_aniso_layer,
            ],
            ui.HCENTER_AL,
        )

        layer_group_w.setLayout(layer_group_l)
        layout.addWidget(layer_group_w)
        ###############################
        ui.add_blank(layout=layout, widget=self)
        ###############################
        folder_group_w, folder_group_l = ui.make_group(
            "Convert folder", l, t, r, b, parent=None
        )

        folder_group_l.addWidget(
            ui.combine_blocks(
                right_or_below=self.btn_label_files,
                left_or_above=self.lbl_label_files,
                min_spacing=70,
            )
        )

        ui.add_widgets(
            folder_group_l,
            [
                self.btn_convert_folder_instance,
                self.btn_convert_folder_semantic,
                self.btn_remove_small_folder,
                self.btn_aniso_folder,
            ],
            ui.HCENTER_AL,
        )

        folder_group_w.setLayout(folder_group_l)
        layout.addWidget(folder_group_w)
        ###############################
        ui.add_blank(layout=layout, widget=self)

        ui.add_widgets(
            layout,
            [
                ui.add_blank(self),
                self.make_close_button(),
                ui.add_blank(self),
                self.lbl_error,
            ],
        )

        ui.ScrollArea.make_scrollable(
            layout, self, min_wh=[230, 400], base_wh=[230, 450]
        )

    def folder_to_semantic(self):
        """Converts folder of labels to semantic labels"""
        if not self.check_ready_folder():
            return

        folder_name = f"converted_to_semantic_labels_{utils.get_date_time()}"

        images = [
            to_semantic(file, is_file_path=True)
            for file in self.labels_filepaths
        ]

        self.save_folder(folder_name, images)

    def layer_to_semantic(self):
        """Converts selected layer to semantic labels"""
        if not self.check_ready_layer():
            return

        im = self._viewer.layers.selection.active.data
        name = self._viewer.layers.selection.active.name
        semantic_labels = to_semantic(im)

        self.save_layer(
            f"{name}_semantic_{utils.get_time_filepath()}"
            + self.filetype_choice.currentText(),
            semantic_labels,
        )

        self._viewer.add_labels(semantic_labels, name=f"converted_semantic")

    def folder_to_instance(self):
        """Converts the chosen folder to instance labels"""
        if not self.check_ready_folder():
            return

        images = [
            to_instance(file, is_file_path=True)
            for file in self.labels_filepaths
        ]

        self.save_folder(
            f"converted_to_instance_labels_{utils.get_date_time()}", images
        )

    def layer_to_instance(self):
        """Converts the selected layer to instance labels"""
        if not self.check_ready_layer():
            return

        im = [self._viewer.layers.selection.active.data]
        name = self._viewer.layers.selection.active.name
        instance_labels = to_instance(im)

        self.save_layer(
            f"{name}_instance_{utils.get_time_filepath()}"
            + self.filetype_choice.currentText(),
            instance_labels,
        )

        self._viewer.add_labels(instance_labels, name=f"converted_instance")

    def layer_remove_small(self):
        """Removes small objects in selected layer"""
        if not self.check_ready_layer():
            return

        im = self._viewer.layers.selection.active.data
        name = self._viewer.layers.selection.active.name

        cleared_labels = clear_small_objects(
            im, self.small_object_thresh_choice.value()
        )

        self.save_layer(
            f"{name}_cleared_{utils.get_time_filepath()}"
            + self.filetype_choice.currentText(),
            cleared_labels,
        )

        self._viewer.add_image(cleared_labels, name=f"small_cleared")

    def folder_remove_small(self):
        """Removes small objects in folder of labels"""
        if not self.check_ready_folder():
            return

        images = [
            clear_small_objects(
                file,
                self.small_object_thresh_choice.value(),
                is_file_path=True,
            )
            for file in self.labels_filepaths
        ]

        self.save_folder(f"small_cleared_{utils.get_date_time()}", images)

    def layer_anisotropy(self):
        """Corrects anisotropy in the currently selected image"""
        if not self.check_ready_layer():
            return

        name = self._viewer.layers.selection.active.name
        zoom_factor = self.anisotropy_converter.get_anisotropy_resolution_zyx()

        vol = np.array(
            self._viewer.layers.selection.active.data, dtype=np.int16
        )
        isotropic_image = utils.resize(vol, zoom_factor)

        self.save_layer(
            f"{name}_isotropic_{utils.get_time_filepath()}"
            + self.filetype_choice.currentText(),
            isotropic_image,
        )

        self._viewer.add_image(isotropic_image, name=f"isotropic")

    def folder_anisotropy(self):
        """Removes anisotropy in folder of images or labels"""
        if not self.check_ready_folder():
            return

        zoom_factor = self.anisotropy_converter.get_anisotropy_resolution_zyx()
        images = [
            utils.resize(imread(file), zoom_factor)
            for file in self.labels_filepaths
        ]

        self.save_folder(f"isotropic_{utils.get_date_time()}", images)

    def check_ready_folder(self):  # TODO add color change
        """Check if results and source folders are correctly set"""
        if self.results_path == "":
            err = "ERROR : please set results folder"
            print(err)
            self.lbl_error.setText(err)
            self.lbl_error.setVisible(True)
            return False
        if self.labels_filepaths != [""]:
            self.lbl_error.setVisible(False)
            return True

        err = "ERROR : please set valid source labels folder"
        print(err)
        self.lbl_error.setText(err)
        self.lbl_error.setVisible(True)
        return False

    def check_ready_layer(self):  # TODO add color change
        """Check if results and layer are selected"""
        if self.results_path == "":
            err = "ERROR : please set results folder"
            print(err)
            self.lbl_error.setText(err)
            self.lbl_error.setVisible(True)
            return False
        if self._viewer.layers.selection.active is None:
            err = "ERROR : Please select a single layer"
            print(err)
            self.lbl_error.setText(err)
            self.lbl_error.setVisible(True)
            return False
        self.lbl_error.setVisible(False)
        return True

    def save_layer(self, file_name, image):

        path = os.path.join(self.results_path, file_name)
        print(self.results_path)
        print(path)

        if self.results_path != "":
            imwrite(
                path,
                image,
            )

    def save_folder(self, folder_name, images):

        results_folder = os.path.join(
            self.results_path,
            folder_name,
        )

        os.makedirs(results_folder, exist_ok=False)

        for file, image in zip(self.labels_filepaths, images):

            path = os.path.join(results_folder, os.path.basename(file))

            imwrite(
                path,
                image,
            )
