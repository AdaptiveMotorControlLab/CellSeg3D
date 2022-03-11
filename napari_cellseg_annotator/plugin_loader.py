import os
from pathlib import Path
import napari
import warnings
import numpy as np
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QSizePolicy,
    QLabel,
    QFileDialog,
    QLineEdit,
    QCheckBox,
)


from skimage import io
from napari_cellseg_annotator import utils
from napari_cellseg_annotator.napari_view_simple import launch_viewers



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

class Loader(QWidget):
    def __init__(self, parent: "napari.viewer.Viewer"):
        super().__init__()
        # self.master = parent
        self._viewer = parent
        self.opath = ""
        self.modpath = ""
        self.btn1 = QPushButton("Open", self)
        self.btn1.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn1.clicked.connect(self.show_dialog_o)
        self.btn2 = QPushButton("Open", self)
        self.btn2.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn2.clicked.connect(self.show_dialog_mod)

        self.textbox = QLineEdit(self)
        self.textbox.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.checkBox = QCheckBox("Create new dataset?")
        self.checkBox.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.btn4 = QPushButton("Start reviewing", self)
        self.btn4.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn4.clicked.connect(self.launch_napari)
        self.btn4.clicked.connect(self.close)
        self.btnb = QPushButton("Close", self)
        self.btnb.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btnb.clicked.connect(self.close)
        self.lbl = QLabel("Images directory", self)
        self.lbl2 = QLabel("Labels directory", self)
        self.lbl4 = QLabel("Model name", self)
        #####################################################################
        # TODO
        self.btntest = QPushButton("test", self)
        self.btntest.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btntest.clicked.connect(self.run_test)
        #####################################################################
        self.build()

    def build(self):
        vbox = QVBoxLayout()
        vbox.addWidget(utils.combine_blocks(self.btn1, self.lbl))
        vbox.addWidget(utils.combine_blocks(self.btn2, self.lbl2))
        vbox.addWidget(utils.combine_blocks(self.textbox, self.lbl4))
        vbox.addWidget(self.checkBox)
        vbox.addWidget(self.btn4)
        vbox.addWidget(self.btnb)
        ##################################################################
        # TODO : remove once done
        test = True
        if test:
            vbox.addWidget(self.btntest)
        ##################################################################
        self.setLayout(vbox)
        self.show()

    def show_dialog_o(self):
        default_path = max(self.opath, self.modpath, os.path.expanduser("~"))
        f_name = QFileDialog.getExistingDirectory(self, "Open directory", default_path)
        if f_name:
            self.opath = f_name
            self.lbl.setText(self.opath)

    def show_dialog_mod(self):
        default_path = max(self.opath, self.modpath, os.path.expanduser("~"))
        f_name = QFileDialog.getExistingDirectory(self, "Open directory", default_path)
        if f_name:
            self.modpath = f_name
            self.lbl2.setText(self.modpath)

    def close(self):
        # self.master.setCurrentIndex(0)
        self._viewer.window.remove_dock_widget(self)

    def launch_napari(self):
        images = utils.load_images(self.opath)
        if self.modpath == "":  # saves empty images of the same size as original images
            labels = np.zeros_like(images.compute())  # dask to numpy
            self.modpath = os.path.join(
                os.path.dirname(self.opath), self.textbox.text()
            )
            os.makedirs(self.modpath, exist_ok=True)
            filenames = [
                fn.name for fn in sorted(list(Path(self.opath).glob("./*png")))
            ]
            for i in range(len(labels)):
                io.imsave(
                    os.path.join(self.modpath, str(i).zfill(4) + ".png"), labels[i]
                )
        else:
            labels = utils.load_saved_masks(self.modpath)
        try:
            labels_raw = utils.load_raw_masks(self.modpath + "_raw")
        except:
            labels_raw = None
            # TODO: viewer argument ?
        view1 = launch_viewers(
            self._viewer,
            images,
            labels,
            labels_raw,
            self.modpath,
            self.textbox.text(),
            self.checkBox.isChecked(),
        )

        # global view_l
        # view_l.close()  # why does it not close the window ??  #TODO use  self.close() ?
        # self.close
        return view1

    ########################
    # TODO : remove once done
    def run_test(self):
        tif = False

        self.opath = "C:/Users/Cyril/Desktop/Proj_bachelor/data/visual_png/sample"
        self.modpath = (
            "C:/Users/Cyril/Desktop/Proj_bachelor/data/visual_png/sample_labels"
        )
        if tif:
            self.opath = "C:/Users/Cyril/Desktop/Proj_bachelor/data/visual_tif/volumes"
            self.modpath = "C:/Users/Cyril/Desktop/Proj_bachelor/data/visual_tif/labels"
        self.launch_napari()
        self.close()

    ########################
