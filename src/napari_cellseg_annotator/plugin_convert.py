import os

from tifffile import imwrite

import napari_cellseg_annotator.interface as ui
from napari_cellseg_annotator import utils
from napari_cellseg_annotator.model_instance_seg import (
    to_semantic,
    to_instance,
)
from napari_cellseg_annotator.plugin_base import BasePluginFolder


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

        self.btn_convert_folder_semantic = ui.make_button(
            "Convert to semantic labels", func=self.folder_to_semantic
        )
        self.btn_convert_layer_semantic = ui.make_button(
            "Convert to semantic labels", func=self.layer_to_semantic
        )
        self.btn_convert_layer_instance = ui.make_button(
            "Convert to instance labels", func=self.layer_to_instance
        )
        self.btn_convert_folder_instance = ui.make_button(
            "Convert to instance labels", func=self.folder_to_instance
        )

        self.btn_image_files.setVisible(False)
        self.lbl_image_files.setVisible(False)

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

        w, layout = ui.make_container_widget()

        results_widget = ui.combine_blocks(
            second=self.btn_result_path,
            first=self.lbl_result_path,
            min_spacing=70,
        )

        ui.make_group(
            "Results",
            solo_dict={"widget": results_widget, "layout": layout},
            L=3,
            T=11,
            R=3,
            B=3,
        )

        folder_group_w, folder_group_l = ui.make_group(
            "Convert folder", l, t, r, b
        )

        folder_group_l.addWidget(
            ui.combine_blocks(
                second=self.btn_label_files,
                first=self.lbl_label_files,
                min_spacing=70,
            ),
            alignment=ui.LEFT_AL,
        )
        folder_group_l.addWidget(
            self.btn_convert_folder_instance, alignment=ui.HCENTER_AL
        )
        folder_group_l.addWidget(
            self.btn_convert_folder_semantic, alignment=ui.HCENTER_AL
        )

        folder_group_w.setLayout(folder_group_l)
        layout.addWidget(folder_group_w)

        layer_group_w, layer_group_l = ui.make_group(
            "Convert selected layer", l, t, r, b
        )

        layer_group_l.addWidget(
            self.btn_convert_layer_instance, alignment=ui.HCENTER_AL
        )
        layer_group_l.addWidget(
            self.btn_convert_layer_semantic, alignment=ui.HCENTER_AL
        )

        layer_group_w.setLayout(layer_group_l)
        layout.addWidget(layer_group_w)

        ui.add_blank(layout=layout, widget=self)
        layout.addWidget(self.make_close_button())

        ui.make_scrollable(layout, self, min_wh=[120, 100], base_wh=[120, 150])

    def folder_to_semantic(self):
        """Converts folder of labels to semantic labels"""

        results_folder = (
            self.results_path
            + f"/converted_to_semantic_labels_{utils.get_date_time()}"
        )

        os.makedirs(results_folder, exist_ok=False)

        for file in self.labels_filepaths:

            image = to_semantic(file, is_file_path=True)

            imwrite(
                results_folder + "/" + os.path.basename(file),
                image,
            )

    def layer_to_semantic(self):
        """Converts selected layer to semantic labels"""

        im = self._viewer.layers.selection.active.data
        name = self._viewer.layers.selection.active.name
        semantic_labels = to_semantic(im)

        if self.results_path != "":
            imwrite(
                self.results_path
                + f"/{name}_semantic_{utils.get_time_filepath()}"
                + self.filetype_choice.currentText(),
                semantic_labels,
            )

        self._viewer.add_labels(semantic_labels, name=f"converted_semantic")

    def folder_to_instance(self):
        """Converts the chosen folder to instance labels"""

        results_folder = (
            self.results_path
            + f"/converted_to_instance_labels_{utils.get_date_time()}"
        )

        os.makedirs(results_folder, exist_ok=False)

        for file in self.labels_filepaths:

            image = to_instance(file, is_file_path=True)

            imwrite(
                results_folder + "/" + os.path.basename(file),
                image,
            )

    def layer_to_instance(self):
        """Converts the selected layer to instance labels"""

        im = [self._viewer.layers.selection.active.data]
        name = self._viewer.layers.selection.active.name
        instance_labels = to_instance(im)

        if self.results_path != "":
            imwrite(
                self.results_path
                + f"/{name}_instance_{utils.get_time_filepath()}"
                + self.filetype_choice.currentText(),
                instance_labels,
            )

        self._viewer.add_labels(instance_labels, name=f"converted_instance")
