import os
from pathlib import Path

import napari
import numpy as np
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
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


class Helper(QWidget):
    # widget testing
    def __init__(self, parent: "napari.viewer.Viewer"):
        super().__init__()
        # self.master = parent
        self.help_url = "https://github.com/C-Achard/cellseg-annotator-test/tree/main"
        self.about_url = (
            "https://wysscenter.ch/advances/3d-computer-vision-for-brain-analysis"
        )
        self._viewer = parent
        self.btn1 = QPushButton("Help...", self)
        # self.btn1.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn1.clicked.connect(lambda: utils.open_url(self.help_url))
        self.btn2 = QPushButton("About...", self)
        # self.btn2.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn2.clicked.connect(lambda: utils.open_url(self.about_url))
        self.btnc = QPushButton("Close", self)
        # self.btnc.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btnc.clicked.connect(self.close)
        self.build()

    def build(self):
        vbox = QVBoxLayout()
        vbox.addWidget(self.btn1)
        vbox.addWidget(self.btn2)
        vbox.addWidget(self.btnc)
        self.setLayout(vbox)
        self.show()

    def close(self):
        self._viewer.window.remove_dock_widget(self)


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
        self.btnb = QPushButton("Close", self)
        self.btnb.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btnb.clicked.connect(self.close)
        self.lbl = QLabel("Images directory", self)
        self.lbl2 = QLabel("Labels directory", self)
        self.lbl4 = QLabel("Model name", self)
        self.build()

    def build(self):
        vbox = QVBoxLayout()
        vbox.addWidget(combine_blocks(self.btn1, self.lbl))
        vbox.addWidget(combine_blocks(self.btn2, self.lbl2))
        vbox.addWidget(combine_blocks(self.textbox, self.lbl4))
        vbox.addWidget(self.checkBox)
        vbox.addWidget(self.btn4)
        vbox.addWidget(self.btnb)

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
        # view_l.close()  # why does it not close the window ??  #TODO use  close()
        self.close
        return view1


def combine_blocks(block1, block2):
    temp_widget = QWidget()
    temp_layout = QHBoxLayout()
    temp_layout.addWidget(block2)
    temp_layout.addWidget(block1)
    temp_widget.setLayout(temp_layout)
    return temp_widget


#
# if __name__ == '__main__':
#     with napari.gui_qt():
#         view_l = napari.Viewer()
#         launcher = App()
#         view_l.window.add_dock_widget(launcher, area='right')
#     # napari.run()
