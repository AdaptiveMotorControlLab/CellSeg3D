import os
import shutil
from pathlib import Path

import napari
import pandas as pd
from qtpy.QtWidgets import QCheckBox
from qtpy.QtWidgets import QLabel
from qtpy.QtWidgets import QPushButton
from qtpy.QtWidgets import QSizePolicy
from qtpy.QtWidgets import QVBoxLayout
from qtpy.QtWidgets import QWidget

from napari_cellseg_annotator import utils


class Predicter(QWidget):
    def __init__(self, viewer: "napari.viewer.Viewer", parent=None):
        super().__init__(parent)
        # self.master = parent
        self._viewer = viewer
        self.opath = ""
        self.labelpath = ""
        self.modelpath = ""
        self.outpath = ""
        self._default_path = [
            self.opath,
            self.labelpath,
            self.modelpath,
            self.outpath,
        ]
        self.btn1 = QPushButton("Open", self)
        self.btn1.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn1.clicked.connect(self.show_dialog_o)
        self.btn2 = QPushButton("Open", self)
        self.btn2.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn2.clicked.connect(self.show_dialog_label)
        self.btn3 = QPushButton("Open", self)
        self.btn3.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn3.clicked.connect(self.show_dialog_model)
        self.btn4 = QPushButton("Open", self)
        self.btn4.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn4.clicked.connect(self.show_dialog_outdir)

        self.checkBox = QCheckBox("Use TAP (Three-Axis-Prediction)")
        self.checkBox.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.checkBox.toggle()

        self.btn5 = QPushButton("Predict", self)
        self.btn5.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn5.clicked.connect(self.predicter)
        self.btnb = QPushButton("Close", self)
        self.btnb.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btnb.clicked.connect(self.close)
        self.lbl = QLabel("Original directory", self)
        self.lbl2 = QLabel("Label directory", self)
        self.lbl3 = QLabel("Model directory (contains model.hdf5)", self)
        self.lbl4 = QLabel("Output directory", self)
        self.build()

        self.model = None
        self.worker_pred = None

    def build(self):
        vbox = QVBoxLayout()
        vbox.addWidget(utils.combine_blocks(self.btn1, self.lbl))
        vbox.addWidget(utils.combine_blocks(self.btn2, self.lbl2))
        vbox.addWidget(utils.combine_blocks(self.btn3, self.lbl3))
        vbox.addWidget(utils.combine_blocks(self.btn4, self.lbl4))
        vbox.addWidget(self.checkBox)
        vbox.addWidget(self.btn5)
        vbox.addWidget(self.btnb)

        self.setLayout(vbox)
        self.show()

    def show_dialog_o(self):

        f_name = utils.open_file_dialog(self, self._default_path)

        if f_name:
            self.opath = f_name
            self.lbl.setText(self.opath)

    def show_dialog_label(self):
        default_path = [self.opath, self.labelpath]
        f_name = utils.open_file_dialog(self, self._default_path)
        if f_name:
            self.labelpath = f_name
            self.lbl2.setText(self.labelpath)

    def show_dialog_model(self):
        default_path = [self.opath, self.labelpath]

        f_name = utils.open_file_dialog(self, self._default_path)
        if f_name:
            self.modelpath = f_name
            self.lbl3.setText(self.modelpath)

    def show_dialog_outdir(self):
        f_name = utils.open_file_dialog(self, self._default_path)

        if f_name:
            self.outpath = f_name
            self.lbl4.setText(self.outpath)

    def close(self):
        # self.master.setCurrentIndex(0)
        self._viewer.window.remove_dock_widget(self)

    def get_newest_csv(self):
        csvs = sorted(list(Path(self.labelpath).glob("./*csv")))
        try:
            csv = pd.read_csv(str(csvs[-1]), index_col=0)
        except:
            csv = None
        return csv, str(csvs[-1])

    def predicter(self):
        ori_imgs, ori_filenames = utils.load_x_gray(self.opath)
        input_shape = (512, 512, 1)
        num_classes = 1

        self.model = get_nested_unet(
            input_shape=input_shape, num_classes=num_classes
        )
        self.model.load_weights(os.path.join(self.modelpath, "model.hdf5"))

        self.btn5.setText("Predicting")

        if self.checkBox.isChecked() is True:
            self.predict(ori_imgs)
        else:
            self.predict_single(ori_imgs)

    def predict(self, ori_imgs):
        try:
            predict_3ax(ori_imgs, self.model, self.outpath)
        except Exception as e:
            print(e)
        if self.labelpath != "":
            try:
                csv, csv_path = self.get_newest_csv()
                if csv:
                    label_names = [
                        node.filename
                        for node in csv.itertuples()
                        if node.train == "Checked"
                    ]
                    for ln in label_names:
                        shutil.copy(
                            os.path.join(self.labelpath, ln),
                            os.path.join(self.outpath, "merged_prediction"),
                        )
                    shutil.copy(
                        str(csv_path),
                        os.path.join(self.outpath, "merged_prediction"),
                    )
            except Exception as e:
                print(e)

        self.btn5.setText("Predict")

    def predict_single(self, ori_imgs):
        try:
            predict_1ax(ori_imgs, self.model, self.outpath)
        except Exception as e:
            print(e)
        if self.labelpath != "":
            try:
                csv, csv_path = self.get_newest_csv()
                if csv:
                    label_names = [
                        node.filename
                        for node in csv.itertuples()
                        if node.train == "Checked"
                    ]
                    for ln in label_names:
                        shutil.copy(
                            os.path.join(self.labelpath, ln),
                            os.path.join(self.outpath, "merged_prediction"),
                        )
                    shutil.copy(
                        str(csv_path),
                        os.path.join(self.outpath, "merged_prediction"),
                    )
            except Exception as e:
                print(e)

        self.btn5.setText("Predict")
