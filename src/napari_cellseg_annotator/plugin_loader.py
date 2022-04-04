import os
import warnings
from pathlib import Path

import napari
import numpy as np
import skimage.io as io
from qtpy import QtGui
from qtpy.QtWidgets import (
    QVBoxLayout,
    QPushButton,
    QSizePolicy,
    QLabel,
    QLineEdit,
    QCheckBox,
)

from napari_cellseg_annotator import utils
from napari_cellseg_annotator.launch_review import launch_review
from napari_cellseg_annotator.plugin_base import BasePlugin

warnings.formatwarning = utils.format_Warning


global_launched_before = False


class Loader(BasePlugin):
    """A plugin for selecting volumes and labels file and launching the review process.
    Inherits from : :doc:`plugin_base`"""

    def __init__(self, viewer: "napari.viewer.Viewer"):
        """Creates a Loader plugin with several buttons :

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

        self.checkBox = QCheckBox("Create new dataset?")
        self.checkBox.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.btn_start = QPushButton("Start reviewing", self)
        self.btn_start.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.btn_start.clicked.connect(self.run_review)

        self.lbl_mod = QLabel("Model name", self)

        self.warn_label = QLabel(
            "WARNING : You already have a review session running.\n"
            "Launching another will close the current one,\n"
            " make sure to save your work beforehand"
        )
        pal = self.warn_label.palette()
        pal.setColor(QtGui.QPalette.WindowText, QtGui.QColor("red"))
        self.warn_label.setPalette(pal)
        #####################################################################
        # TODO remove once done
        self.test_button = True
        if self.test_button:
            self.btntest = QPushButton("test", self)
            self.btntest.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            self.btntest.clicked.connect(self.run_test)
        #####################################################################

        self.build()

    def build(self):
        """Build buttons in a layout and add them to the napari Viewer"""

        vbox = QVBoxLayout()

        global global_launched_before
        if global_launched_before:
            vbox.addWidget(self.warn_label)
            warnings.warn(
                "You already have a review session running.\n"
                "Launching another will close the current one,\n"
                " make sure to save your work beforehand"
            )

        vbox.addWidget(
            utils.combine_blocks(self.filetype_choice, self.file_handling_box)
        )
        self.filetype_choice.setVisible(False)

        vbox.addWidget(utils.combine_blocks(self.btn_image, self.lbl_image))

        vbox.addWidget(utils.combine_blocks(self.btn_label, self.lbl_label))
        # vbox.addWidget(self.lblft2)

        vbox.addWidget(utils.combine_blocks(self.textbox, self.lbl_mod))

        vbox.addWidget(self.checkBox)
        vbox.addWidget(self.btn_start)
        vbox.addWidget(self.btn_close)

        ##################################################################
        # remove once done ?

        if self.test_button:
            vbox.addWidget(self.btntest)
        ##################################################################
        self.setLayout(vbox)
        # self.show()
        # self._viewer.window.add_dock_widget(self, name="Loader", area="right")

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
        images = utils.load_images(
            self.image_path, self.filetype, self.file_handling_box.isChecked()
        )
        if (
            self.label_path == ""
        ):  # saves empty images of the same size as original images
            labels = np.zeros_like(images.compute())  # dask to numpy
            self.label_path = os.path.join(
                os.path.dirname(self.image_path), self.textbox.text()
            )
            os.makedirs(self.label_path, exist_ok=True)
            filenames = [
                fn.name
                for fn in sorted(
                    list(Path(self.image_path).glob("./*" + self.filetype))
                )
            ]
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
                self.file_handling_box.isChecked(),
            )
        try:
            labels_raw = utils.load_raw_masks(
                self.label_path + "_raw", self.filetype
            )
        except:
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
                self.file_handling_box.isChecked(),
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
                self.file_handling_box.isChecked(),
            )
            self._viewer.window.remove_dock_widget(self)
            global_launched_before = True

        return view1

    ########################
    # TODO : remove once done
    def run_test(self):
        self.filetype = self.filetype_choice.currentText()
        if self.file_handling_box.isChecked():
            self.image_path = (
                "C:/Users/Cyril/Desktop/Proj_bachelor/data/visual_png/sample"
            )
            self.label_path = "C:/Users/Cyril/Desktop/Proj_bachelor/data/visual_png/sample_labels"
        else:
            self.image_path = "C:/Users/Cyril/Desktop/Proj_bachelor/data/visual_tif/volumes/images.tif"
            self.label_path = "C:/Users/Cyril/Desktop/Proj_bachelor/data/visual_tif/labels/testing_im.tif"
        self.run_review()
        # self.close()

    ########################
    def close(self):
        """Close widget and remove it from window.
        Sets the check for an active session to false, so that if the user closes manually and doesn't launch the review,
        the active session warning does not display and a new viewer is not opened when launching for the first time.
        """
        global global_launched_before  # if user closes window rather than launching review, does not count as active session
        if global_launched_before:
            global_launched_before = False
        print("close req")
        self._viewer.window.remove_dock_widget(self)
