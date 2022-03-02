import os
from pathlib import Path
import numpy as np
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSizePolicy, QLabel, QFileDialog,
                             QTabWidget, QLineEdit, QCheckBox)
from skimage import io
import utils
from napari_view_simple import launch_viewers



class Loader(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.master = parent
        self.opath = ""
        self.modpath = ""
        self.btn1 = QPushButton('open', self)
        self.btn1.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn1.clicked.connect(self.show_dialog_o)
        self.btn2 = QPushButton('open', self)
        self.btn2.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn2.clicked.connect(self.show_dialog_mod)

        self.textbox = QLineEdit(self)
        self.textbox.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.checkBox = QCheckBox("Create new dataset?")
        self.checkBox.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.btn4 = QPushButton('Launch napari !', self)
        self.btn4.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn4.clicked.connect(self.launch_napari)
        self.btnb = QPushButton('back', self)
        self.btnb.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btnb.clicked.connect(self.back)
        self.lbl = QLabel('Images directory', self)
        self.lbl2 = QLabel('Labels directory', self)
        self.lbl4 = QLabel('Model name', self)
        self.build()

    def build(self):
        vbox = QVBoxLayout()
        vbox.addWidget(QWidget.combine_blocks(self.btn1, self.lbl))
        vbox.addWidget(QWidget.combine_blocks(self.btn2, self.lbl2))
        vbox.addWidget(QWidget.combine_blocks(self.textbox, self.lbl4))
        vbox.addWidget(self.checkBox)
        vbox.addWidget(self.btn4)
        vbox.addWidget(self.btnb)

        self.setLayout(vbox)
        self.show()

    def show_dialog_o(self):
        default_path = max(self.opath, self.modpath, os.path.expanduser('~'))
        f_name = QFileDialog.getExistingDirectory(self, 'Open Directory', default_path)
        if f_name:
            self.opath = f_name
            self.lbl.setText(self.opath)

    def show_dialog_mod(self):
        default_path = max(self.opath, self.modpath, os.path.expanduser('~'))
        f_name = QFileDialog.getExistingDirectory(self, 'Open Directory', default_path)
        if f_name:
            self.modpath = f_name
            self.lbl2.setText(self.modpath)

    def back(self):
        self.master.setCurrentIndex(0)

    def launch_napari(self):
        images = utils.load_images(self.opath)
        if self.modpath == "":  # saves empty images of the same size as original images
            labels = np.zeros_like(images.compute())  # dask to numpy
            self.modpath = os.path.join(os.path.dirname(self.opath), self.textbox.text())
            os.makedirs(self.modpath, exist_ok=True)
            filenames = [fn.name for fn in sorted(list(Path(self.opath).glob('./*png')))]
            for i in range(len(labels)):
                io.imsave(os.path.join(self.modpath, str(i).zfill(4) + '.png'), labels[i])
        else:
            labels = utils.load_saved_masks(self.modpath)
        try:
            labels_raw = utils.load_raw_masks(self.modpath + '_raw')
        except:
            labels_raw = None
        view1 = launch_viewers(images, labels, labels_raw, self.modpath, self.textbox.text(), self.checkBox.isChecked())
        global view_l
        view_l.close()  # why does it not close the window ??
        return view1