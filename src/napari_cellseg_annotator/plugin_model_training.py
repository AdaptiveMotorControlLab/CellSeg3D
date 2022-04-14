import os
import gc
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import napari
import numpy as np
import torch
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.figure import Figure
# MONAI
from monai.data import (
    DataLoader,
    PatchDataset,
    decollate_batch,
    pad_list_data_collate,
)
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
    QWidget,
    QVBoxLayout,
    QPushButton,
    QSizePolicy,
    QLayout,
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
        ######################
        ######################
        # TEST TODO REMOVE
        import glob

        directory = os.path.dirname(os.path.realpath(__file__)) + str(
            Path("/models/dataset/volumes")
        )

        lab_directory = os.path.dirname(os.path.realpath(__file__)) + str(
            Path("/models/dataset/lab_sem")
        )
        self.images_filepaths = sorted(
            glob.glob(os.path.join(directory, "*.tif"))
        )

        self.labels_filepaths = sorted(
            glob.glob(os.path.join(lab_directory, "*.tif"))
        )

        #######################
        #######################
        #######################

        self.results_path = os.path.dirname(os.path.realpath(__file__)) + str(
            Path("/models/saved_weights")
        )

        # default values
        self.num_samples = 2
        self.batch_size = 1
        self.epochs = 4
        self.val_interval = 2

        self.model = None  # TODO : custom model loading ?
        self.worker = None
        self.data = None

        self.loss_dict = {
            "Dice loss": DiceLoss(sigmoid=True),
            "Focal loss": FocalLoss(),
            "Dice-Focal Loss": DiceFocalLoss(sigmoid=True, lambda_dice=0.2),
        }

        self.metric_values = []
        self.epoch_loss_values = []
        self.canvas = None
        self.train_loss_plot = None
        self.dice_metric_plot = None

        ################################
        # interface
        self.epoch_choice = QSpinBox()
        self.epoch_choice.setValue(self.epochs)
        self.epoch_choice.setRange(2, 1000)
        self.epoch_choice.setSingleStep(2)
        self.epoch_choice.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.lbl_epoch_choice = QLabel("Number of epochs : ", self)

        self.loss_choice = QComboBox()
        self.loss_choice.addItems(sorted(self.loss_dict.keys()))
        self.lbl_loss_choice = QLabel("Loss function", self)

        self.sample_choice = QSpinBox()
        self.sample_choice.setValue(self.num_samples)
        self.sample_choice.setRange(2, 50)
        self.sample_choice.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.lbl_sample_choice = QLabel(
            "Number of samples from image : ", self
        )

        self.batch_choice = QSpinBox()
        self.batch_choice.setValue(self.batch_size)
        self.batch_choice.setRange(1, 10)
        self.batch_choice.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.lbl_batch_choice = QLabel("Batch size : ", self)

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

        param_tab = QWidget()

        param_tab_layout = QVBoxLayout()
        param_tab_layout.setSizeConstraint(QLayout.SetFixedSize)

        param_tab_layout.addWidget(
            utils.combine_blocks(self.filetype_choice, self.lbl_filetype)
        )  # file extension

        param_tab_layout.addWidget(
            utils.combine_blocks(self.btn_image_files, self.lbl_image_files)
        )  # volumes
        param_tab_layout.addWidget(
            utils.combine_blocks(self.btn_label_files, self.lbl_label_files)
        )  # labels

        # param_tab_layout.addWidget(
        #     utils.combine_blocks(self.model_choice, self.lbl_model_choice)
        # )  # model choice

        param_tab_layout.addWidget(
            utils.combine_blocks(self.btn_result_path, self.lbl_result_path)
        )  # results folder

        param_tab_layout.addWidget(QLabel("", self))
        param_tab_layout.addWidget(
            utils.combine_blocks(self.model_choice, self.lbl_model_choice)
        )  # model choice
        param_tab_layout.addWidget(
            utils.combine_blocks(self.loss_choice, self.lbl_loss_choice)
        )  # loss choice
        param_tab_layout.addWidget(
            utils.combine_blocks(self.epoch_choice, self.lbl_epoch_choice)
        )  # epochs
        param_tab_layout.addWidget(
            utils.combine_blocks(self.sample_choice, self.lbl_sample_choice)
        )  # number of samples
        param_tab_layout.addWidget(
            utils.combine_blocks(self.batch_choice, self.lbl_batch_choice)
        )  # batch size

        param_tab_layout.addWidget(QLabel("", self))

        param_tab_layout.addWidget(self.btn_start)
        param_tab_layout.addWidget(self.btn_close)
        # TODO : what to train ? predefined model ? custom model ?

        param_tab.setLayout(param_tab_layout)

        self.addTab(param_tab, "Basic parameters")

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

        if not self.check_ready():
            return
        self.num_samples = self.sample_choice.value()
        self.batch_size = self.batch_choice.value()
        self.data = self.create_train_dataset_dict()

        self.btn_close.setVisible(False)


        if self.worker is not None:
            if self.worker.is_running:
                pass
            else:
                self.worker.start()
                self.btn_start.setText("Running... Click to stop")
        else:

            self.worker = self.train()
            self.worker.started.connect(lambda: print("Worker is running..."))
            self.worker.finished.connect(lambda: print("Worker stopped"))
            self.worker.finished.connect(self.exit_train)
            if self.get_device().type == "cuda":
                self.worker.finished.connect(self.empty_cuda_cache)

        if self.worker.is_running:
            print("Stop request, waiting for next validation step...")
            self.btn_start.setText("Stopping...")
            self.worker.quit()
        else:
            # self.worker.start()
            self.btn_start.setText("Running...  Click to stop")

    def exit_train(self):
        self.worker = None
        if self.model is not None:
            del self.model
            self.model = None
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr((obj,'data') and torch.is_tensor(obj.data))):
                    # print(type(obj), obj.size())
                    del obj
            except:
                pass
        gc.collect()
        del self.epoch_loss_values
        del self.metric_values
        del self.data

        self.btn_start.setText("Start")
        self.btn_close.setVisible(True)

        # self.close()
        # del self

    def plot_loss(self, loss, dice_metric):
        with plt.style.context("dark_background"):
            # update loss
            self.train_loss_plot.set_title("Epoch average loss")
            self.train_loss_plot.set_xlabel("Epoch")
            self.train_loss_plot.set_ylabel("Loss")
            x = [i + 1 for i in range(len(loss))]
            y = loss
            self.train_loss_plot.plot(x, y)
            # update metrics
            x = [self.val_interval * (i + 1) for i in range(len(dice_metric))]
            y = dice_metric

            epoch_min = (np.argmax(y) + 1) * self.val_interval
            dice_min = np.max(y)

            self.dice_metric_plot.plot(x, y)
            self.dice_metric_plot.set_title(
                "Validation metric : Mean Dice coefficient"
            )
            self.dice_metric_plot.set_xlabel("Epoch")

            self.dice_metric_plot.scatter(
                epoch_min, dice_min, c="r", label="Maximum Dice coeff."
            )
            self.dice_metric_plot.legend(
                facecolor="#262930", loc="upper right"
            )
            self.canvas.draw_idle()

    def update_loss_plot(self):

        # print(len(self.epoch_loss_values))
        # print(self.epoch_loss_values)
        # print(self.metric_values)

        loss = self.epoch_loss_values
        metric = self.metric_values
        epoch = len(loss)
        if epoch < self.val_interval*2:
            return
        elif epoch == self.val_interval*2:
            bckgrd_color = (0, 0, 0, 0)  # '#262930'
            with plt.style.context("dark_background"):

                self.canvas = FigureCanvas(Figure(figsize=(10, 3)))
                # loss plot
                self.train_loss_plot = self.canvas.figure.add_subplot(1, 2, 1)
                # dice metric validation plot
                self.dice_metric_plot = self.canvas.figure.add_subplot(1, 2, 2)

                self.canvas.figure.set_facecolor(bckgrd_color)
                self.dice_metric_plot.set_facecolor(bckgrd_color)
                self.train_loss_plot.set_facecolor(bckgrd_color)

                # self.canvas.figure.tight_layout()

                self.canvas.figure.subplots_adjust(
                    left=0.1,
                    bottom=0.2,
                    right=0.95,
                    top=0.9,
                    wspace=0.2,
                    hspace=0,
                )

            self.canvas.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

            # tab_index = self.addTab(self.canvas, "Loss plot")
            # self.setCurrentIndex(tab_index)
            self._viewer.window.add_dock_widget(self.canvas, area="bottom")
            self.plot_loss(loss, metric)
        else:
            with plt.style.context("dark_background"):

                self.train_loss_plot.cla()
                self.dice_metric_plot.cla()

                self.plot_loss(loss, metric)

    @thread_worker(connect={"yielded": update_loss_plot})
    def train(self):

        device = self.get_device()
        model_id = self.get_model(self.model_choice.currentText())
        model_name = self.model_choice.currentText()
        data_dicts = self.data
        max_epochs = self.epoch_choice.value()
        loss_function = self.get_loss(self.loss_choice.currentText())
        val_interval = self.val_interval
        batch_size = self.batch_choice.value()
        results_path = self.results_path
        num_samples = self.sample_choice.value()

        model = model_id.get_net()
        model = model.to(device)

        epoch_loss_values = []
        metric_values = []

        # TODO param : % of validation from training set
        train_files, val_files = (
            data_dicts[0 : int(len(data_dicts) * 0.9)],
            data_dicts[int(len(data_dicts) * 0.9) :],
        )
        # print("train/val")
        # print(train_files)
        # print(val_files)
        sample_loader = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                RandSpatialCropSamplesd(
                    keys=["image", "label"],
                    roi_size=(110, 110, 110),
                    max_roi_size=(120, 120, 120),
                    num_samples=num_samples,
                ),
                SpatialPadd(
                    keys=["image", "label"], spatial_size=(128, 128, 128)
                ),
                EnsureTyped(keys=["image", "label"]),
            ]
        )

        train_transforms = Compose(  # TODO : figure out which ones ?
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

        val_transforms = Compose(
            [
                # LoadImaged(keys=["image", "label"]),
                # EnsureChannelFirstd(keys=["image", "label"]),
                EnsureTyped(keys=["image", "label"]),
            ]
        )

        train_ds = PatchDataset(
            data=train_files,
            transform=train_transforms,
            patch_func=sample_loader,
            samples_per_image=num_samples,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=pad_list_data_collate,
        )

        val_ds = PatchDataset(
            data=val_files,
            transform=val_transforms,
            patch_func=sample_loader,
            samples_per_image=num_samples,
        )

        val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=4)

        # TODO : more parameters/flexibility
        post_pred = AsDiscrete(threshold=0.3)
        post_label = EnsureType()

        optimizer = torch.optim.Adam(model.parameters(), 1e-3)
        dice_metric = DiceMetric(include_background=True, reduction="mean")

        best_metric = -1
        best_metric_epoch = -1

        time = utils.get_date_time()

        weights_filename = f"{model_name}_best_metric" + f"_{time}.pth"
        if device.type == "cuda":
            print("\nUsing GPU :")
            print(torch.cuda.get_device_name(0))
        else:
            print("Using CPU")

        for epoch in range(max_epochs):
            print("-" * 10)
            print(f"Epoch {epoch + 1}/{max_epochs}")
            if device.type == "cuda":
                print("Memory Usage:")
                print(
                    "Allocated:",
                    round(torch.cuda.memory_allocated(0) / 1024**3, 1),
                    "GB",
                )
                print(
                    "Cached:   ",
                    round(torch.cuda.memory_reserved(0) / 1024**3, 1),
                    "GB",
                )

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
                epoch_loss += loss.detach().item()
                print(
                    f"{step}/{len(train_ds) // train_loader.batch_size}, "
                    f"Train_loss: {loss.detach().item():.4f}"
                )
            epoch_loss /= step
            self.epoch_loss_values.append(epoch_loss)
            print(f"Epoch {epoch + 1} Average loss: {epoch_loss:.4f}")


            if (epoch + 1) % val_interval == 0:
                model.eval()
                with torch.no_grad():
                    for val_data in val_loader:
                        val_inputs, val_labels = (
                            val_data["image"].to(device),
                            val_data["label"].to(device),
                        )

                        val_outputs = model_id.get_validation(
                            model, val_inputs
                        )

                        pred = decollate_batch(val_outputs)

                        labs = decollate_batch(val_labels)

                        val_outputs = [
                            post_pred(res_tensor) for res_tensor in pred
                        ]

                        val_labels = [
                            post_label(res_tensor) for res_tensor in labs
                        ]

                        dice_metric(y_pred=val_outputs, y=val_labels)

                    metric = dice_metric.aggregate().detach().item()
                    dice_metric.reset()

                    self.metric_values.append(metric)

                    yield self

                    if metric > best_metric:
                        best_metric = metric
                        best_metric_epoch = epoch + 1
                        torch.save(
                            model.state_dict(),
                            os.path.join(results_path, weights_filename),
                        )
                        print("Saved best metric model")
                    print(
                        f"Current epoch: {epoch + 1} Current mean dice: {metric:.4f}"
                        f"\nBest mean dice: {best_metric:.4f} "
                        f"at epoch: {best_metric_epoch}"
                    )
        print("=" * 10)
        print("Done !")
        print(
            f"Train completed, best_metric: {best_metric:.4f} "
            f"at epoch: {best_metric_epoch}"
        )

        # self.close()

    def close(self):
        """Close the widget"""
        self._viewer.window.remove_dock_widget(self)
