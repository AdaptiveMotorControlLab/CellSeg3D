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


global_launched_before = False


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

        self.btn_start = ui.make_button(
            "Start reviewing", self.run_review, self
        )

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

        global global_launched_before
        if global_launched_before:
            layout.addWidget(self.warn_label)
            warnings.warn(
                "You already have a review session running.\n"
                "Launching another will close the current one,\n"
                " make sure to save your work beforehand"
            )

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

        ui.make_scrollable(
            contained_layout=layout, containing_widget=tab, min_wh=[190, 300]
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

        self.filetype = self.filetype_choice.currentText()
        self.as_folder = self.file_handling_box.isChecked()
        if self.anisotropy_widgets.checkbox.isChecked():
            zoom = utils.anisotropy_zoom_factor(
                self.anisotropy_widgets.get_anisotropy_factors()
            )
            # reordering
            zoom = [zoom[2], zoom[1], zoom[0]]
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

        global global_launched_before
        if global_launched_before:
            new_viewer = napari.Viewer()
            view1 = launch_review(
                new_viewer,
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
            warnings.warn(
                "Opening several loader sessions in one window is not supported; opening in new window"
            )
            self._viewer.close()
        else:
            viewer = self._viewer
            print("new sess")
            view1 = launch_review(
                viewer,
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
            self.remove_from_viewer()

            global_launched_before = True

        return view1

    def remove_from_viewer(self):
        """Close widget and remove it from window.
        Sets the check for an active session to false, so that if the user closes manually and doesn't launch the review,
        the active session warning does not display and a new viewer is not opened when launching for the first time.
        """
        global global_launched_before  # if user closes window rather than launching review, does not count as active session
        if global_launched_before:
            global_launched_before = False
        # print("close req")
        try:
            self._viewer.window.remove_dock_widget(self)
        except LookupError:
            return
