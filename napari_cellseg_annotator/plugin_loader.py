import os
import warnings
from pathlib import Path
import napari
import numpy as np
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QSizePolicy,
    QLabel,
    QFileDialog,
    QComboBox,
    QLineEdit,
    QCheckBox,
)
from skimage import io
from napari_cellseg_annotator import utils
from napari_cellseg_annotator.launch_review import launch_review


def format_Warning(message, category, filename, lineno, line=""):
    return (
        str(filename)
        + ":"
        + str(lineno)
        + ": "
        + category.__name__
        + ": "
        + str(message)
        + "\n"
    )


warnings.formatwarning = format_Warning


global_launched_before = False


class Loader(QWidget):
    """A plugin for selecting volumes and labels file and launching the review process."""

    def __init__(self, viewer: "napari.viewer.Viewer", parent=None):
        """Creates a Loader plugin with several buttons :

        Open file prompt to select volumes directory

        Open file prompt to select labels directory

        A dropdown menu with a choice of png or tif filetypes

        A checkbox if you want to create a new status csv for the dataset

        A button to launch the review process (see :doc:`launch_review`)
        """

        super().__init__(parent)

        # self.master = parent
        self._viewer = viewer
        """napari.viewer.Viewer: viewer in which the widget is displayed"""

        self.opath = ""
        """str: path to output folder"""
        self.modpath = ""
        """str: path to mask folder"""
        self.filetype = ""
        """str: filetype, .tif or .png"""

        self.btn1 = QPushButton("Open", self)
        self.btn1.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn1.clicked.connect(self.show_dialog_o)
        self.btn2 = QPushButton("Open", self)
        self.btn2.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn2.clicked.connect(self.show_dialog_mod)

        self.filetype_choice = QComboBox()
        self.filetype_choice.addItems([".png", ".tif"])
        self.filetype_choice.setSizePolicy(
            QSizePolicy.Fixed, QSizePolicy.Fixed
        )

        self.textbox = QLineEdit(self)
        self.textbox.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.checkBox = QCheckBox("Create new dataset?")
        self.checkBox.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.btn4 = QPushButton("Start reviewing", self)
        self.btn4.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.btn4.clicked.connect(self.run_review)
        # self.btn4.clicked.connect(self.close)
        self.btnb = QPushButton("Close", self)
        self.btnb.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btnb.clicked.connect(self.close)
        self.lbl = QLabel("Images directory", self)
        self.lbl2 = QLabel("Labels directory", self)
        self.lbl4 = QLabel("Model name", self)
        #####################################################################
        # TODO remove once done
        self.btntest = QPushButton("test", self)
        self.lblft = QLabel("Filetype :", self)
        self.lblft2 = QLabel("(Folders of .png or single .tif files)")
        self.btntest.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btntest.clicked.connect(self.run_test)
        #####################################################################

        self.build()

    def build(self):
        """Build buttons in a layout and add them to the napari Viewer"""

        vbox = QVBoxLayout()

        vbox.addWidget(utils.combine_blocks(self.btn1, self.lbl))

        vbox.addWidget(utils.combine_blocks(self.btn2, self.lbl2))
        vbox.addWidget(self.lblft2)
        vbox.addWidget(utils.combine_blocks(self.filetype_choice, self.lblft))

        vbox.addWidget(utils.combine_blocks(self.textbox, self.lbl4))

        vbox.addWidget(self.checkBox)
        vbox.addWidget(self.btn4)
        vbox.addWidget(self.btnb)
        ##################################################################
        # remove once done ?
        test_button = True
        if test_button:
            vbox.addWidget(self.btntest)
        ##################################################################
        self.setLayout(vbox)
        self.show()

    def show_dialog_o(self):

        default_path = [self.opath, self.modpath]
        f_name = utils.open_file_dialog(self, default_path)

        if f_name:
            self.opath = f_name
            self.lbl.setText(self.opath)

    def show_dialog_mod(self):
        default_path = [self.opath, self.modpath]
        f_name = utils.open_file_dialog(self, default_path)

        if f_name:
            self.modpath = f_name
            self.lbl2.setText(self.modpath)

    def close(self):
        """Close the widget"""
        # self.master.setCurrentIndex(0)
        self._viewer.window.remove_dock_widget(self)

    def run_review(self):

        """Launches review process by loading the files from the chosen folders,
        and adds several widgets to the napari Viewer.
        If the review process has been launched once before,
        closes the window entirely and launches the review process in a fresh window.

        TODO:

        * Add warning that launching again will close the current window and lose all progress

        * Save work done before leaving

        See :doc:`launch_review`

        Returns:
            napari.viewer.Viewer: self.viewer
        """
        self.filetype = self.filetype_choice.currentText()
        images = utils.load_images(self.opath, self.filetype)
        if (
            self.modpath == ""
        ):  # saves empty images of the same size as original images
            labels = np.zeros_like(images.compute())  # dask to numpy
            self.modpath = os.path.join(
                os.path.dirname(self.opath), self.textbox.text()
            )
            os.makedirs(self.modpath, exist_ok=True)
            filenames = [
                fn.name
                for fn in sorted(
                    list(Path(self.opath).glob("./*" + self.filetype))
                )
            ]
            for i in range(len(labels)):
                io.imsave(
                    os.path.join(
                        self.modpath, str(i).zfill(4) + self.filetype
                    ),
                    labels[i],
                )
        else:
            labels = utils.load_saved_masks(self.modpath, self.filetype)
        try:
            labels_raw = utils.load_raw_masks(
                self.modpath + "_raw", self.filetype
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
                self.modpath,
                self.textbox.text(),
                self.checkBox.isChecked(),
                self.filetype,
            )
            warnings.warn(
                "WARNING : Opening several loader sessions in one window is not supported; opening in new window"
            )
            self._viewer.close()
        else:
            new_viewer = self._viewer

            view1 = launch_review(
                new_viewer,
                images,
                labels,
                labels_raw,
                self.modpath,
                self.textbox.text(),
                self.checkBox.isChecked(),
                self.filetype,
            )
            global_launched_before = True
            self.close()

        return view1

    ########################
    # TODO : remove once done
    def run_test(self):
        self.filetype = self.filetype_choice.currentText()

        self.opath = (
            "C:/Users/Cyril/Desktop/Proj_bachelor/data/visual_png/sample"
        )
        self.modpath = "C:/Users/Cyril/Desktop/Proj_bachelor/data/visual_png/sample_labels"
        if self.filetype == ".tif":
            self.opath = (
                "C:/Users/Cyril/Desktop/Proj_bachelor/data/visual_tif/volumes"
            )
            self.modpath = (
                "C:/Users/Cyril/Desktop/Proj_bachelor/data/visual_tif/labels"
            )
        self.run_review()
        # self.close()

    ########################
