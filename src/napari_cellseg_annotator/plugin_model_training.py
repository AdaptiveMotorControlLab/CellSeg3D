import os
import warnings
from pathlib import Path

import napari
import torch
from monai.data import (
    DataLoader,
    PatchDataset,
    decollate_batch,
    pad_list_data_collate,
)
# MONAI
from monai.losses import DiceLoss, FocalLoss, DiceFocalLoss
from monai.metrics import DiceMetric
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    EnsureType,
    EnsureTyped,
    RandSpatialCropSamplesd,
    SpatialPadd,
    RandShiftIntensityd,
    Rand3DElasticd,
)
from napari.qt.threading import thread_worker
# Qt
from qtpy.QtWidgets import (
    QVBoxLayout,
    QPushButton,
    QSizePolicy,
    QLabel,
    QComboBox,
    QSpinBox,
)

from napari_cellseg_annotator import utils
from napari_cellseg_annotator.model_framework import ModelFramework


# TODO : setup training + check #param entries to add more flexibility/advanced options


class Trainer(ModelFramework):
    def __init__(self, viewer: "napari.viewer.Viewer"):

        super().__init__(viewer)

        # self.master = parent
        self._viewer = viewer
        """napari.viewer.Viewer: viewer in which the widget is displayed"""

        ######################
        # TEST TODO REMOVE
        import glob
        directory = "C:/Users/Cyril/Desktop/Proj_bachelor/data/visual_tif/volumes"
        lab_directory = "C:/Users/Cyril/Desktop/Proj_bachelor/data/visual_tif/lab_sem"
        self.images_filepaths = sorted(glob.glob(os.path.join(directory, "*.tif")))

        self.labels_filepaths = sorted(glob.glob(os.path.join(lab_directory, "*.tif")))

        #######################

        self.results_path = os.path.dirname(os.path.realpath(__file__)) + str(
            Path("/models/saved_weights")
        )

        self.num_samples = 2

        self.batch_size = 1
        self.epochs = 4

        self.model = None  # TODO : model loading ?
        self.worker = None

        self.loss_dict = {
            "Dice loss": DiceLoss(sigmoid=True),
            "Focal loss": FocalLoss(),
            "Dice-Focal Loss": DiceFocalLoss(sigmoid=True, lambda_dice=0.2),
        }

        ################################
        # interface
        self.epoch_choice = QSpinBox()
        self.epoch_choice.setValue(self.epochs)
        self.epoch_choice.setRange(1, 100)
        self.epoch_choice.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.lbl_epoch_choice = QLabel("Number of epochs : ", self)

        self.loss_choice = QComboBox()
        self.loss_choice.addItems(sorted(self.loss_dict.keys()))
        self.lbl_loss_choice = QLabel("Loss function", self)

        self.sample_choice = QSpinBox()
        self.sample_choice.setValue(self.num_samples)
        self.sample_choice.setRange(1, 50)
        self.sample_choice.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.lbl_sample_choice = QLabel(
            "Number of samples from image : ", self
        )

        self.btn_start = QPushButton("Start training")
        self.btn_start.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn_start.clicked.connect(self.start)

        self.btn_model_path.setVisible(False)
        self.lbl_model_path.setVisible(False)

        self.build()

    def check_ready(self):
        if self.images_filepaths != [] and self.labels_filepaths != []:
            return True
        else:
            warnings.formatwarning = utils.format_Warning
            warnings.warn("Image and label paths are not correctly set")
            return False

    def build(self):

        vbox = QVBoxLayout()

        vbox.addWidget(
            utils.combine_blocks(self.filetype_choice, self.lbl_filetype)
        )  # file extension

        vbox.addWidget(
            utils.combine_blocks(self.btn_image_files, self.lbl_image_files)
        )  # volumes
        vbox.addWidget(
            utils.combine_blocks(self.btn_label_files, self.lbl_label_files)
        )  # labels

        # vbox.addWidget(
        #     utils.combine_blocks(self.model_choice, self.lbl_model_choice)
        # )  # model choice

        vbox.addWidget(
            utils.combine_blocks(self.btn_result_path, self.lbl_result_path)
        )  # results folder

        vbox.addWidget(QLabel("", self))
        vbox.addWidget(
            utils.combine_blocks(self.model_choice, self.lbl_model_choice)
        )  # model choice
        vbox.addWidget(
            utils.combine_blocks(self.loss_choice, self.lbl_loss_choice)
        )  # loss choice
        vbox.addWidget(
            utils.combine_blocks(self.epoch_choice, self.lbl_epoch_choice)
        )  # epochs
        vbox.addWidget(
            utils.combine_blocks(self.sample_choice, self.lbl_sample_choice)
        )  # number of samples

        vbox.addWidget(QLabel("", self))

        vbox.addWidget(self.btn_start)
        vbox.addWidget(self.btn_close)
        # TODO : what to train ? predefined model ? custom model ?

        self.setLayout(vbox)

    def show_dialog_lab(self):
        f_name = utils.open_file_dialog(self, self._default_path)

        if f_name:
            self.label_path = f_name
            self.lbl_label.setText(self.label_path)

    def show_dialog_dat(self):
        f_name = utils.open_file_dialog(self, self._default_path)

        if f_name:
            self.data_path = f_name
            self.lbl_dat.setText(self.label_path)

    def start(self):

        self.num_samples = self.sample_choice.value()

        self.sample_loader = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                RandSpatialCropSamplesd(
                    keys=["image", "label"],
                    roi_size=(110, 110, 110),
                    max_roi_size=(120, 120, 120),
                    num_samples=self.num_samples,
                ),
                SpatialPadd(
                    keys=["image", "label"], spatial_size=(128, 128, 128)
                ),
                EnsureTyped(keys=["image", "label"]),
            ]
        )

        self.train_transforms = Compose(  # TODO : figure out which ones ?
            [
                RandShiftIntensityd(keys=["image"], offsets=0.7),
                Rand3DElasticd(
                    keys=["image", "label"],
                    sigma_range=(0.3, 0.7),
                    magnitude_range=(0.3, 0.7),
                ),
                EnsureTyped(keys=["image", "label"]),
            ]
        )

        self.val_transforms = Compose(
            [
                # LoadImaged(keys=["image", "label"]),
                #EnsureChannelFirstd(keys=["image", "label"]),
                EnsureTyped(keys=["image", "label"]),
            ]
        )
        self.btn_close.setVisible(False)

        # TODO : multithreading ?
        if self.worker is not None :
            if self.worker.is_running :
                pass
            else :
                self.worker.start()
                self.btn_start.setText("Stop")
        else :
            self.worker = self.train()
            self.worker.started.connect(lambda: print("Worker is running..."))
            self.worker.finished.connect(lambda: print("Worker stopped"))

        if self.worker.is_running:
            print("Stop training requested")
            self.model.stop_training = True
            self.btn_start.setText("Start training")
            self.btn_close.setVisible(True)
            self.worker = None
        else:
            self.worker.start()
            self.btn_start.setText("Stop")

    @thread_worker
    def train(self):

        device = self.get_device()

        data_dicts = self.create_train_dataset_dict()

        # TODO param : % of validation from training set
        train_files, val_files = data_dicts, data_dicts
        # print("train/val")
        # print(train_files)
        # print(val_files)

        train_ds = PatchDataset(
            data=train_files,
            transform=self.train_transforms,
            patch_func=self.sample_loader,
            samples_per_image=self.num_samples,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=pad_list_data_collate,
        )

        val_ds = PatchDataset(
            data=val_files,
            transform=self.val_transforms,
            patch_func=self.sample_loader,
            samples_per_image=self.num_samples,
        )

        val_loader = DataLoader(
            val_ds, batch_size=self.batch_size, num_workers=4
        )

        model_id = self.get_model(self.model_choice.currentText())
        model = model_id.get_net()
        model = model.to(device)

        # TODO : more parameters/flexibility
        post_pred = AsDiscrete(threshold=0.3)
        post_label = EnsureType()
        val_interval = 2
        max_epochs = self.epochs
        loss_function = self.get_loss(self.loss_choice.currentText())
        optimizer = torch.optim.Adam(model.parameters(), 1e-3)
        dice_metric = DiceMetric(include_background=True, reduction="mean")

        best_metric = -1
        best_metric_epoch = -1
        epoch_loss_values = []
        metric_values = []

        time = utils.get_date_time()
        weights_filename = "best_model" + f"_{time}_.pth"

        for epoch in range(max_epochs):
            print("-" * 10)
            print(f"Epoch {epoch + 1}/{max_epochs}")
            model.train()
            epoch_loss = 0
            step = 0
            for batch_data in train_loader:
                step += 1
                inputs, labels = (
                    batch_data["image"].to(device),
                    batch_data["label"].to(device),
                )
                optimizer.zero_grad()
                outputs = model_id.get_output(model, inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                print(
                    f"{step}/{len(train_ds) // train_loader.batch_size}, "
                    f"Train_loss: {loss.item():.4f}"
                )
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            print(f"Epoch {epoch + 1} Average loss: {epoch_loss:.4f}")

            if (epoch + 1) % val_interval == 0:
                model.eval()
                with torch.no_grad():
                    for val_data in val_loader:
                        val_inputs, val_labels = (
                            val_data["image"].to(device),
                            val_data["label"].to(device),
                        )


                        val_outputs = model_id.get_validation(model, val_inputs)

                        pred = decollate_batch(val_outputs)

                        labs = decollate_batch(val_labels)


                        val_outputs = [
                            post_pred(res_tensor) for res_tensor in pred
                        ]

                        val_labels = [
                            post_label(res_tensor) for res_tensor in labs
                        ]

                        dice_metric(y_pred=val_outputs, y=val_labels)

                    metric = dice_metric.aggregate().item()
                    dice_metric.reset()

                    metric_values.append(metric)
                    if metric > best_metric:
                        best_metric = metric
                        best_metric_epoch = epoch + 1
                        torch.save(
                            model.state_dict(),
                            os.path.join(self.results_path, weights_filename),
                        )
                        print("Saved best metric model")
                    print(
                        f"Current epoch: {epoch + 1} Current mean dice: {metric:.4f}"
                        f"\nBest mean dice: {best_metric:.4f} "
                        f"at epoch: {best_metric_epoch}"
                    )
        print(
            f"Train completed, best_metric: {best_metric:.4f} "
            f"at epoch: {best_metric_epoch}"
        )
        #self.close()
        self.btn_start.setText("Start training")
        self.btn_close.setVisible(True)

    def close(self):
        """Close the widget"""
        self._viewer.window.remove_dock_widget(self)
