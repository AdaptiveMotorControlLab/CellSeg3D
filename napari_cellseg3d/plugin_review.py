import os
import warnings

import napari
import numpy as np
import pims
import skimage.io as io

# Qt
from qtpy.QtWidgets import QLineEdit
from qtpy.QtWidgets import QSizePolicy

# local
from napari_cellseg3d import interface as ui
from napari_cellseg3d import utils
from napari_cellseg3d.launch_review import launch_review
from napari_cellseg3d.plugin_base import BasePluginSingleImage

warnings.formatwarning = utils.format_Warning


class Reviewer(BasePluginSingleImage):
    """A plugin for selecting volumes and labels file and launching the review process.
    Inherits from : :doc:`plugin_base`"""

    def __init__(self, viewer: "napari.viewer.Viewer"):
        """Creates a Reviewer plugin with several buttons :

        * Open file prompt to select volumes directory

        * Open file prompt to select labels directory

        * A dropdown menu with a choice of png or tif filetypes

        * A checkbox if you want to create a new status csv for the dataset

        * A button to launch the review process (see :doc:`launch_review`)
        """

        super().__init__(viewer)

        # self._viewer = viewer

        self.textbox = QLineEdit(self)
        self.textbox.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.checkBox = ui.make_checkbox("Create new dataset ?")

        self.btn_start = ui.Button("Start reviewing", self.run_review, self)

        self.lbl_mod = ui.make_label("Name", self)

        self.warn_label = ui.make_label(
            "WARNING : You already have a review session running.\n"
            "Launching another will close the current one,\n"
            " make sure to save your work beforehand",
            None,
        )

        self.anisotropy_widgets = ui.AnisotropyWidgets(
            self, default_x=1.5, default_y=1.5, default_z=5
        )

        ###########################
        # tooltips
        self.textbox.setToolTip("Name of the csv results file")
        self.checkBox.setToolTip(
            "Ignore any pre-existing csv with the specified name and create a new one"
        )
        ###########################

        self.build()

    def build(self):
        """Build buttons in a layout and add them to the napari Viewer"""

        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.MinimumExpanding)

        tab, layout = ui.make_container(0, 0, 1, 1)

        # ui.add_blank(self, layout)
        ###########################
        data_group_w, data_group_l = ui.make_group("Data")

        ui.add_widgets(
            data_group_l,
            [
                ui.combine_blocks(
                    self.filetype_choice,
                    self.file_handling_box,
                    horizontal=False,
                ),
                ui.combine_blocks(self.btn_image, self.lbl_image),
                ui.combine_blocks(self.btn_label, self.lbl_label),
            ],
        )

        self.filetype_choice.setVisible(False)

        data_group_w.setLayout(data_group_l)
        layout.addWidget(data_group_w)
        ###########################
        ui.add_blank(self, layout)
        ###########################
        ui.add_to_group("Image parameters", self.anisotropy_widgets, layout)
        ###########################
        ui.add_blank(self, layout)
        ###########################
        csv_param_w, csv_param_l = ui.make_group("CSV parameters")

        ui.add_widgets(
            csv_param_l,
            [
                ui.combine_blocks(
                    self.textbox,
                    self.lbl_mod,
                    horizontal=False,
                    l=5,
                    t=0,
                    r=5,
                    b=5,
                ),
                self.checkBox,
            ],
        )

        csv_param_w.setLayout(csv_param_l)
        layout.addWidget(csv_param_w)
        ###########################
        ui.add_blank(self, layout)
        ###########################

        ui.add_widgets(layout, [self.btn_start, self.btn_close])

        ui.ScrollArea.make_scrollable(
            contained_layout=layout, parent=tab, min_wh=[190, 300]
        )

        self.addTab(tab, "Review")

        self.setMinimumSize(180, 100)
        # self.show()
        # self._viewer.window.add_dock_widget(self, name="Reviewer", area="right")

    def run_review(self):

        """Launches review process by loading the files from the chosen folders,
        and adds several widgets to the napari Viewer.
        If the review process has been launched once before,
        closes the window entirely and launches the review process in a fresh window.

        TODO:

        * Save work done before leaving

        See :doc:`launch_review`

        Returns:
            napari.viewer.Viewer: self.viewer
        """

        self.reset()

        self.filetype = self.filetype_choice.currentText()
        self.as_folder = self.file_handling_box.isChecked()
        if self.anisotropy_widgets.is_enabled():
            zoom = self.anisotropy_widgets.get_anisotropy_resolution_zyx(
                as_factors=True
            )
        else:
            zoom = [1, 1, 1]

        images = utils.load_images(
            self.image_path, self.filetype, self.as_folder
        )
        if (
            self.label_path == ""  # TODO check if it works
        ):  # saves empty images of the same size as original images
            if self.as_folder:
                labels = np.zeros_like(images.compute())  # dask to numpy
            self.label_path = os.path.join(
                os.path.dirname(self.image_path), self.textbox.text()
            )
            os.makedirs(self.label_path, exist_ok=True)

            for i in range(len(labels)):
                io.imsave(
                    os.path.join(
                        self.label_path, str(i).zfill(4) + self.filetype
                    ),
                    labels[i],
                )
        else:
            labels = utils.load_saved_masks(
                self.label_path,
                self.filetype,
                self.as_folder,
            )
        try:
            labels_raw = utils.load_raw_masks(
                self.label_path + "_raw", self.filetype
            )
        except pims.UnknownFormatError:
            labels_raw = None
        except FileNotFoundError:
            # TODO : might not work, test with predi labels later
            labels_raw = None

        print("New review session\n" + "*" * 20)
        previous_viewer = self._viewer
        self._viewer, self.docked_widgets = launch_review(
            images,
            labels,
            labels_raw,
            self.label_path,
            self.textbox.text(),
            self.checkBox.isChecked(),
            self.filetype,
            self.as_folder,
            zoom,
        )
        previous_viewer.close()

    def reset(self):
        self._viewer.layers.clear()
        self.remove_docked_widgets()
