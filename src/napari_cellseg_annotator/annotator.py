import os
import warnings
from pathlib import Path

import napari
import numpy as np
from qtpy.QtWidgets import QCheckBox
from qtpy.QtWidgets import QComboBox
from qtpy.QtWidgets import QFileDialog
from qtpy.QtWidgets import QLabel
from qtpy.QtWidgets import QLineEdit
from qtpy.QtWidgets import QPushButton
from qtpy.QtWidgets import QSizePolicy
from qtpy.QtWidgets import QVBoxLayout
from qtpy.QtWidgets import QWidget
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


class Helper(QWidget):
    # widget testing
    def __init__(self, parent: "napari.viewer.Viewer"):
        super().__init__()
        # self.master = parent
        self.help_url = (
            "https://github.com/C-Achard/cellseg-annotator-test/tree/main"
        )
        self.about_url = "https://wysscenter.ch/advances/3d-computer-vision-for-brain-analysis"
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


global_launched_before = False


class Loader(QWidget):
    def __init__(self, parent: "napari.viewer.Viewer"):
        super(Loader, self).__init__()

        # self.master = parent
        self._viewer = parent
        self.opath = ""
        self.modpath = ""
        self.filetype = ""

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
        # TODO : remove once done
        test = True
        if test:
            vbox.addWidget(self.btntest)
        ##################################################################
        self.setLayout(vbox)
        self.show()

    def show_dialog_o(self):
        default_path = max(self.opath, self.modpath, os.path.expanduser("~"))
        f_name = QFileDialog.getExistingDirectory(
            self, "Open directory", default_path
        )
        if f_name:
            self.opath = f_name
            self.lbl.setText(self.opath)

    def show_dialog_mod(self):
        default_path = max(self.opath, self.modpath, os.path.expanduser("~"))
        f_name = QFileDialog.getExistingDirectory(
            self, "Open directory", default_path
        )
        if f_name:
            self.modpath = f_name
            self.lbl2.setText(self.modpath)

    def close(self):
        # self.master.setCurrentIndex(0)
        self._viewer.window.remove_dock_widget(self)

    def run_review(self):

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
            # global view_l
        # view_l.close()  # why does it not close the window ??  #use self.close() ?

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


# class Trainer(QWidget):
#     def __init__(self, parent: "napari.viewer.Viewer"):
#         super().__init__()
#         # self.master = parent
#         self._viewer = parent
#         self.opath = ""
#         self.labelpath = ""
#         self.modelpath = ""
#         self.btn1 = QPushButton("Open", self)
#         self.btn1.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
#         self.btn1.clicked.connect(self.show_dialog_o)
#         self.btn2 = QPushButton("Open", self)
#         self.btn2.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
#         self.btn2.clicked.connect(self.show_dialog_label)
#         self.btn3 = QPushButton("Open", self)
#         self.btn3.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
#         self.btn3.clicked.connect(self.show_dialog_model)
#
#         self.btn4 = QPushButton("Start training", self)
#         self.btn4.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
#         self.btn4.clicked.connect(self.trainer)
#         self.btnb = QPushButton("Close", self)
#         self.btnb.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
#         self.btnb.clicked.connect(self.close)
#         self.lbl = QLabel("Original directory", self)
#         self.lbl2 = QLabel("Label directory", self)
#         self.lbl3 = QLabel("Model output directory", self)
#         self.build()
#
#         self.model = None
#         self.worker = None
#         self.worker2 = None
#
#     def build(self):
#         vbox = QVBoxLayout()
#         vbox.addWidget(utils.combine_blocks(self.btn1, self.lbl))
#         vbox.addWidget(utils.combine_blocks(self.btn2, self.lbl2))
#         vbox.addWidget(utils.combine_blocks(self.btn3, self.lbl3))
#         vbox.addWidget(self.btn4)
#         vbox.addWidget(self.btnb)
#
#         self.setLayout(vbox)
#         # self.show()
#
#     def show_dialog_o(self):
#         default_path = max(self.opath, self.labelpath, os.path.expanduser("~"))
#         f_name = QFileDialog.getExistingDirectory(self, "Open directory", default_path)
#         if f_name:
#             self.opath = f_name
#             self.lbl.setText(self.opath)
#
#     def show_dialog_label(self):
#         default_path = max(self.opath, self.labelpath, os.path.expanduser("~"))
#         f_name = QFileDialog.getExistingDirectory(self, "Open directory", default_path)
#         if f_name:
#             self.labelpath = f_name
#             self.lbl2.setText(self.labelpath)
#
#     def show_dialog_model(self):
#         default_path = max(self.opath, self.labelpath, os.path.expanduser("~"))
#         f_name = QFileDialog.getExistingDirectory(self, "Open directory", default_path)
#         if f_name:
#             self.modelpath = f_name
#             self.lbl3.setText(self.modelpath)
#
#     def close(self):
#         # self.master.setCurrentIndex(0)
#         self._viewer.window.remove_dock_widget(self)
#
#     def get_newest_csv(self):
#         csvs = sorted(list(Path(self.labelpath).glob("./*csv")))
#         csv = pd.read_csv(str(csvs[-1]), index_col=0)
#         return csv
#
#     def update_layer(self, df):
#         plt.figure(figsize=(10, 5))
#         plt.subplot(1, 2, 1)
#         plt.plot(list(df["epoch"]), list(df["dice_coeff"]), label="dice_coeff")
#         plt.xlim(0, 400)
#         plt.ylim(0, 1)
#         plt.legend()
#         plt.subplot(1, 2, 2)
#         plt.plot(list(df["epoch"]), list(df["loss"]), label="loss")
#         plt.legend()
#         plt.xlim(0, 400)
#         buf = IO.BytesIO()
#         plt.savefig(buf, format="png")
#         buf.seek(0)
#         im = Image.open(buf)
#         im = np.array(im)
#         buf.close()
#         try:
#             view_l.layers["result"].data = im
#         except KeyError:
#             view_l.add_image(im, name="result")
#
#     def trainer(self):
#         if self.worker:
#             if self.worker.is_running:
#                 pass
#             else:
#                 self.worker.start()
#                 self.btn4.setText("Stop")
#         else:
#             ori_imgs, ori_filenames = utils.load_x_gray(self.opath)
#             label_imgs, label_filenames = utils.load_Y_gray(
#                 self.labelpath, normalize=False
#             )
#             train_csv = self.get_newest_csv()
#             train_ori_imgs, train_label_imgs = utils.select_train_data(
#                 dataframe=train_csv,
#                 ori_imgs=ori_imgs,
#                 label_imgs=label_imgs,
#                 ori_filenames=ori_filenames,
#             )
#             devided_train_ori_imgs = utils.divide_imgs(train_ori_imgs)
#             devided_train_label_imgs = utils.divide_imgs(train_label_imgs)
#             devided_train_label_imgs = np.where(
#                 devided_train_label_imgs < 0, 0, devided_train_label_imgs
#             )
#
#             self.model = get_nested_unet(input_shape=(512, 512, 1), num_classes=1)
#
#             self.worker = self.train(
#                 devided_train_ori_imgs, devided_train_label_imgs, self.model
#             )
#             self.worker.started.connect(lambda: print("worker is running..."))
#             self.worker.finished.connect(lambda: print("worker stopped"))
#             self.worker2 = self.yield_csv()
#             self.worker2.yielded.connect(self.update_layer)
#             self.worker2.start()
#
#         if self.worker.is_running:
#             self.model.stop_training = True
#             print("Stop training requested")
#             self.btn4.setText("Start training")
#             self.worker = None
#         else:
#             self.worker.start()
#             self.btn4.setText("Stop")
#
#     @thread_worker
#     def train(self, devided_train_ori_imgs, devided_train_label_imgs, model):
#         train_unet(
#             X_train=devided_train_ori_imgs,
#             Y_train=devided_train_label_imgs,
#             csv_path=os.path.join(self.modelpath, "train_log.csv"),
#             model_path=os.path.join(self.modelpath, "model.hdf5"),
#             model=model,
#         )
#
#     @thread_worker
#     def yield_csv(self):
#         while True:
#             df = pd.read_csv(os.path.join(self.modelpath, "train_log.csv"))
#             df["epoch"] = df["epoch"] + 1
#             yield df
#             time.sleep(30)


# class Predicter(QWidget):
#     def __init__(self, parent: "napari.viewer.Viewer"):
#         super().__init__()
#         # self.master = parent
#         self._viewer = parent
#         self.opath = ""
#         self.labelpath = ""
#         self.modelpath = ""
#         self.outpath = ""
#         self.btn1 = QPushButton("Open", self)
#         self.btn1.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
#         self.btn1.clicked.connect(self.show_dialog_o)
#         self.btn2 = QPushButton("Open", self)
#         self.btn2.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
#         self.btn2.clicked.connect(self.show_dialog_label)
#         self.btn3 = QPushButton("Open", self)
#         self.btn3.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
#         self.btn3.clicked.connect(self.show_dialog_model)
#         self.btn4 = QPushButton("Open", self)
#         self.btn4.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
#         self.btn4.clicked.connect(self.show_dialog_outdir)
#
#         self.checkBox = QCheckBox(
#             "Check the box if you want to use TAP (Three-Axis-Prediction"
#         )
#         self.checkBox.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
#         self.checkBox.toggle()
#
#         self.btn5 = QPushButton("Predict", self)
#         self.btn5.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
#         self.btn5.clicked.connect(self.predicter)
#         self.btnb = QPushButton("Close", self)
#         self.btnb.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
#         self.btnb.clicked.connect(self.close)
#         self.lbl = QLabel("Original directory", self)
#         self.lbl2 = QLabel("Label directory", self)
#         self.lbl3 = QLabel("Model directory (contains model.hdf5)", self)
#         self.lbl4 = QLabel("Output directory", self)
#         self.build()
#
#         self.model = None
#         self.worker_pred = None
#
#     def build(self):
#         vbox = QVBoxLayout()
#         vbox.addWidget(utils.combine_blocks(self.btn1, self.lbl))
#         vbox.addWidget(utils.combine_blocks(self.btn2, self.lbl2))
#         vbox.addWidget(utils.combine_blocks(self.btn3, self.lbl3))
#         vbox.addWidget(utils.combine_blocks(self.btn4, self.lbl4))
#         vbox.addWidget(self.checkBox)
#         vbox.addWidget(self.btn5)
#         vbox.addWidget(self.btnb)
#
#         self.setLayout(vbox)
#         self.show()
#
#     def show_dialog_o(self):
#         default_path = max(self.opath, self.labelpath, os.path.expanduser("~"))
#         f_name = QFileDialog.getExistingDirectory(self, "Open directory", default_path)
#         if f_name:
#             self.opath = f_name
#             self.lbl.setText(self.opath)
#
#     def show_dialog_label(self):
#         default_path = max(self.opath, self.labelpath, os.path.expanduser("~"))
#         f_name = QFileDialog.getExistingDirectory(self, "Open directory", default_path)
#         if f_name:
#             self.labelpath = f_name
#             self.lbl2.setText(self.labelpath)
#
#     def show_dialog_model(self):
#         default_path = max(self.opath, self.labelpath, os.path.expanduser("~"))
#         f_name = QFileDialog.getExistingDirectory(self, "Open directory", default_path)
#         if f_name:
#             self.modelpath = f_name
#             self.lbl3.setText(self.modelpath)
#
#     def show_dialog_outdir(self):
#         default_path = max(self.opath, self.labelpath, os.path.expanduser("~"))
#         f_name = QFileDialog.getExistingDirectory(self, "Open directory", default_path)
#         if f_name:
#             self.outpath = f_name
#             self.lbl4.setText(self.outpath)
#
#     def close(self):
#         # self.master.setCurrentIndex(0)
#         self._viewer.window.remove_dock_widget(self)
#
#     def get_newest_csv(self):
#         csvs = sorted(list(Path(self.labelpath).glob("./*csv")))
#         try:
#             csv = pd.read_csv(str(csvs[-1]), index_col=0)
#         except:
#             csv = None
#         return csv, str(csvs[-1])
#
#     def predicter(self):
#         ori_imgs, ori_filenames = utils.load_x_gray(self.opath)
#         input_shape = (512, 512, 1)
#         num_classes = 1
#
#         self.model = get_nested_unet(input_shape=input_shape, num_classes=num_classes)
#         self.model.load_weights(os.path.join(self.modelpath, "model.hdf5"))
#
#         self.btn5.setText("Predicting")
#
#         if self.checkBox.isChecked() is True:
#             self.predict(ori_imgs)
#         else:
#             self.predict_single(ori_imgs)
#
#     def predict(self, ori_imgs):
#         try:
#             predict_3ax(ori_imgs, self.model, self.outpath)
#         except Exception as e:
#             print(e)
#         if self.labelpath != "":
#             try:
#                 csv, csv_path = self.get_newest_csv()
#                 if csv:
#                     label_names = [
#                         node.filename
#                         for node in csv.itertuples()
#                         if node.train == "Checked"
#                     ]
#                     for ln in label_names:
#                         shutil.copy(
#                             os.path.join(self.labelpath, ln),
#                             os.path.join(self.outpath, "merged_prediction"),
#                         )
#                     shutil.copy(
#                         str(csv_path), os.path.join(self.outpath, "merged_prediction")
#                     )
#             except Exception as e:
#                 print(e)
#
#         self.btn5.setText("Predict")
#
#     def predict_single(self, ori_imgs):
#         try:
#             predict_1ax(ori_imgs, self.model, self.outpath)
#         except Exception as e:
#             print(e)
#         if self.labelpath != "":
#             try:
#                 csv, csv_path = self.get_newest_csv()
#                 if csv:
#                     label_names = [
#                         node.filename
#                         for node in csv.itertuples()
#                         if node.train == "Checked"
#                     ]
#                     for ln in label_names:
#                         shutil.copy(
#                             os.path.join(self.labelpath, ln),
#                             os.path.join(self.outpath, "merged_prediction"),
#                         )
#                     shutil.copy(
#                         str(csv_path), os.path.join(self.outpath, "merged_prediction")
#                     )
#             except Exception as e:
#                 print(e)
#
#         self.btn5.setText("Predict")


# class Entrance(QWidget):
#     def __init__(self, parent):
#         super().__init__()
#         #self.master = parent
#         self.btn1 = QPushButton('Loader', self)
#         # self.btn1.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
#         self.btn1.clicked.connect(self.move_to_loader)
#         # self.btn2 = QPushButton('Trainer', self)
#         # # self.btn2.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
#         # self.btn2.clicked.connect(self.move_to_trainer)
#         # self.btn3 = QPushButton('Predicter', self)
#         # # self.btn3.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
#         # self.btn3.clicked.connect(self.move_to_predicter)
#         self.build()
#
#     def build(self):
#         vbox = QVBoxLayout()
#         vbox.addWidget(self.btn1)
#         # vbox.addWidget(self.btn2)
#         # vbox.addWidget(self.btn3)
#
#         self.setLayout(vbox)
#         self.show()
#
#     def move_to_loader(self):
#         self.master.setCurrentIndex(1)
#
#     # def move_to_trainer(self):
#     #     self.master.setCurrentIndex(2)
#     #
#     # def move_to_predicter(self):
#     #     self.master.setCurrentIndex(3)


# class App(QTabWidget):
# def __init__(self):
#     super().__init__()
#     self.setWindowTitle("napari launcher")
#     self.tab1 = Entrance(self)
#     self.tab2 = Loader(self)
# self.tab3 = Trainer(self)
# self.tab4 = Predicter(self)

# add to tab page
# self.addTab(self.tab1, "Entrance")
# self.addTab(self.tab2, "Loader")
# self.addTab(self.tab3, "Trainer")
# self.addTab(self.tab4, "Predicter")

# self.setStyleSheet("QTabWidget::pane { border: 0; }")
# self.tabBar().hide()
# self.resize(500, 400)


#
# if __name__ == '__main__':
#     with napari.gui_qt():
#         view_l = napari.Viewer()
#         launcher = App()
#         view_l.window.add_dock_widget(launcher, area='right')
#     # napari.run()
