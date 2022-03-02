import os
import time
from pathlib import Path
import io as IO
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSizePolicy, QLabel, QFileDialog,
                             QTabWidget, QLineEdit, QCheckBox)
from napari.qt import thread_worker
import utils
from models import get_nested_unet
from train import train_unet

class Trainer(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.master = parent
        self.opath = ""
        self.labelpath = ""
        self.modelpath = ""
        self.btn1 = QPushButton('open', self)
        self.btn1.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn1.clicked.connect(self.show_dialog_o)
        self.btn2 = QPushButton('open', self)
        self.btn2.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn2.clicked.connect(self.show_dialog_label)
        self.btn3 = QPushButton('open', self)
        self.btn3.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn3.clicked.connect(self.show_dialog_model)

        self.btn4 = QPushButton('start training', self)
        self.btn4.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn4.clicked.connect(self.trainer)
        self.btnb = QPushButton('back', self)
        self.btnb.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btnb.clicked.connect(self.back)
        self.lbl = QLabel('original dir', self)
        self.lbl2 = QLabel('label dir', self)
        self.lbl3 = QLabel('model output dir', self)
        self.build()

        self.model = None
        self.worker = None
        self.worker2 = None

    def build(self):
        vbox = QVBoxLayout()
        vbox.addWidget(QWidget.combine_blocks(self.btn1, self.lbl))
        vbox.addWidget(QWidget.combine_blocks(self.btn2, self.lbl2))
        vbox.addWidget(QWidget.combine_blocks(self.btn3, self.lbl3))
        vbox.addWidget(self.btn4)
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

    def back(self):
        self.master.setCurrentIndex(0)

    def get_newest_csv(self):
        csvs = sorted(list(Path(self.labelpath).glob('./*csv')))
        csv = pd.read_csv(str(csvs[-1]), index_col=0)
        return csv

    def update_layer(self, df):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(list(df['epoch']), list(df['dice_coeff']), label='dice_coeff')
        plt.xlim(0, 400)
        plt.ylim(0, 1)
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(list(df['epoch']), list(df['loss']), label='loss')
        plt.legend()
        plt.xlim(0, 400)
        buf = IO.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        im = Image.open(buf)
        im = np.array(im)
        buf.close()
        try:
            view_l.layers['result'].data = im
        except KeyError:
            view_l.add_image(
                im, name='result'
            )

    def trainer(self):
        if self.worker:
            if self.worker.is_running:
                pass
            else:
                self.worker.start()
                self.btn4.setText('stop')
        else:
            ori_imgs, ori_filenames = utils.load_X_gray(self.opath)
            label_imgs, label_filenames = utils.load_Y_gray(self.labelpath, normalize=False)
            train_csv = self.get_newest_csv()
            train_ori_imgs, train_label_imgs = utils.select_train_data(
                dataframe=train_csv,
                ori_imgs=ori_imgs,
                label_imgs=label_imgs,
                ori_filenames=ori_filenames,
            )
            devided_train_ori_imgs = utils.divide_imgs(train_ori_imgs)
            devided_train_label_imgs = utils.divide_imgs(train_label_imgs)
            devided_train_label_imgs = np.where(
                devided_train_label_imgs < 0,
                0,
                devided_train_label_imgs
            )

            self.model = get_nested_unet(input_shape=(512, 512, 1), num_classes=1)

            self.worker = self.train(devided_train_ori_imgs, devided_train_label_imgs, self.model)
            self.worker.started.connect(lambda: print("worker is running..."))
            self.worker.finished.connect(lambda: print("worker stopped"))
            self.worker2 = self.yield_csv()
            self.worker2.yielded.connect(self.update_layer)
            self.worker2.start()

        if self.worker.is_running:
            self.model.stop_training = True
            print("stop training requested")
            self.btn4.setText('start training')
            self.worker = None
        else:
            self.worker.start()
            self.btn4.setText('stop')

    @thread_worker
    def train(self, devided_train_ori_imgs, devided_train_label_imgs, model):
        train_unet(
            X_train=devided_train_ori_imgs,
            Y_train=devided_train_label_imgs,
            csv_path=os.path.join(self.modelpath, "train_log.csv"),
            model_path=os.path.join(self.modelpath, "model.hdf5"),
            model=model
        )

    @thread_worker
    def yield_csv(self):
        while True:
            df = pd.read_csv(os.path.join(self.modelpath, "train_log.csv"))
            df['epoch'] = df['epoch'] + 1
            yield df
            time.sleep(30)