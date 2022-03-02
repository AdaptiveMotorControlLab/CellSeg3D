import os
import shutil
from pathlib import Path
import pandas as pd
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSizePolicy, QLabel, QFileDialog,
                             QTabWidget, QLineEdit, QCheckBox)
import utils
from models import get_nested_unet
from predict import predict_3ax, predict_1ax

class Predicter(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.master = parent
        self.opath = ""
        self.labelpath = ""
        self.modelpath = ""
        self.outpath = ""
        self.btn1 = QPushButton('open', self)
        self.btn1.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn1.clicked.connect(self.show_dialog_o)
        self.btn2 = QPushButton('open', self)
        self.btn2.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn2.clicked.connect(self.show_dialog_label)
        self.btn3 = QPushButton('open', self)
        self.btn3.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn3.clicked.connect(self.show_dialog_model)
        self.btn4 = QPushButton('open', self)
        self.btn4.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn4.clicked.connect(self.show_dialog_outdir)

        self.checkBox = QCheckBox("Check the box if you want to use TAP (Three-Axis-Prediction")
        self.checkBox.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.checkBox.toggle()

        self.btn5 = QPushButton('predict', self)
        self.btn5.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn5.clicked.connect(self.predicter)
        self.btnb = QPushButton('back', self)
        self.btnb.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btnb.clicked.connect(self.back)
        self.lbl = QLabel('original dir', self)
        self.lbl2 = QLabel('label dir', self)
        self.lbl3 = QLabel('model dir (contains model.hdf5)', self)
        self.lbl4 = QLabel('output dir', self)
        self.build()

        self.model = None
        self.worker_pred = None

    def build(self):
        vbox = QVBoxLayout()
        vbox.addWidget(QWidget.combine_blocks(self.btn1, self.lbl))
        vbox.addWidget(QWidget.combine_blocks(self.btn2, self.lbl2))
        vbox.addWidget(QWidget.combine_blocks(self.btn3, self.lbl3))
        vbox.addWidget(QWidget.combine_blocks(self.btn4, self.lbl4))
        vbox.addWidget(self.checkBox)
        vbox.addWidget(self.btn5)
        vbox.addWidget(self.btnb)

        self.setLayout(vbox)
        self.show()

    def show_dialog_o(self):
        default_path = max(self.opath, self.labelpath, os.path.expanduser('~'))
        f_name = QFileDialog.getExistingDirectory(self, 'Open Directory', default_path)
        if f_name:
            self.opath = f_name
            self.lbl.setText(self.opath)

    def show_dialog_label(self):
        default_path = max(self.opath, self.labelpath, os.path.expanduser('~'))
        f_name = QFileDialog.getExistingDirectory(self, 'Open Directory', default_path)
        if f_name:
            self.labelpath = f_name
            self.lbl2.setText(self.labelpath)

    def show_dialog_model(self):
        default_path = max(self.opath, self.labelpath, os.path.expanduser('~'))
        f_name = QFileDialog.getExistingDirectory(self, 'Open Directory', default_path)
        if f_name:
            self.modelpath = f_name
            self.lbl3.setText(self.modelpath)

    def show_dialog_outdir(self):
        default_path = max(self.opath, self.labelpath, os.path.expanduser('~'))
        f_name = QFileDialog.getExistingDirectory(self, 'Open Directory', default_path)
        if f_name:
            self.outpath = f_name
            self.lbl4.setText(self.outpath)

    def back(self):
        self.master.setCurrentIndex(0)

    def get_newest_csv(self):
        csvs = sorted(list(Path(self.labelpath).glob('./*csv')))
        try:
            csv = pd.read_csv(str(csvs[-1]), index_col=0)
        except:
            csv = None
        return csv, str(csvs[-1])

    def predicter(self):
        ori_imgs, ori_filenames = utils.load_X_gray(self.opath)
        input_shape = (512, 512, 1)
        num_classes = 1

        self.model = get_nested_unet(input_shape=input_shape, num_classes=num_classes)
        self.model.load_weights(os.path.join(self.modelpath, "model.hdf5"))

        self.btn5.setText('predicting')

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
                    label_names = [node.filename for node in csv.itertuples() if node.train == "Checked"]
                    for ln in label_names:
                        shutil.copy(os.path.join(self.labelpath, ln), os.path.join(self.outpath, 'merged_prediction'))
                    shutil.copy(str(csv_path), os.path.join(self.outpath, 'merged_prediction'))
            except Exception as e:
                print(e)

        self.btn5.setText('predict')

    def predict_single(self, ori_imgs):
        try:
            predict_1ax(ori_imgs, self.model, self.outpath)
        except Exception as e:
            print(e)
        if self.labelpath != "":
            try:
                csv, csv_path = self.get_newest_csv()
                if csv:
                    label_names = [node.filename for node in csv.itertuples() if node.train == "Checked"]
                    for ln in label_names:
                        shutil.copy(os.path.join(self.labelpath, ln), os.path.join(self.outpath, 'merged_prediction'))
                    shutil.copy(str(csv_path), os.path.join(self.outpath, 'merged_prediction'))
            except Exception as e:
                print(e)

        self.btn5.setText('predict')