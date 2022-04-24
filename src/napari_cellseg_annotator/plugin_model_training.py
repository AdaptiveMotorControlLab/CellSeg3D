import os
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
from monai.data import DataLoader
from monai.data import PatchDataset
from monai.data import decollate_batch
from monai.data import pad_list_data_collate
from monai.losses import DiceCELoss
from monai.losses import DiceFocalLoss
from monai.losses import DiceLoss
from monai.losses import FocalLoss
from monai.losses import GeneralizedDiceLoss
from monai.losses import TverskyLoss
from monai.metrics import DiceMetric
from monai.transforms import Compose
from monai.transforms import EnsureChannelFirstd
from monai.transforms import EnsureType
from monai.transforms import EnsureTyped
from monai.transforms import LoadImaged
from monai.transforms import Orientationd
from monai.transforms import Rand3DElasticd
from monai.transforms import RandAffined
from monai.transforms import RandFlipd
from monai.transforms import RandRotate90d
from monai.transforms import RandShiftIntensityd
from monai.transforms import RandSpatialCropSamplesd
from monai.transforms import SpatialPadd
from napari.qt.threading import thread_worker
# Qt
from qtpy.QtWidgets import QComboBox
from qtpy.QtWidgets import QLabel
from qtpy.QtWidgets import QLayout
from qtpy.QtWidgets import QProgressBar
from qtpy.QtWidgets import QPushButton
from qtpy.QtWidgets import QSizePolicy
from qtpy.QtWidgets import QSpinBox
from qtpy.QtWidgets import QVBoxLayout
from qtpy.QtWidgets import QWidget

from napari_cellseg_annotator import utils
from napari_cellseg_annotator.model_framework import ModelFramework


class Trainer(ModelFramework):
    """A plugin to train pre-defined Pytorch models for one-channel segmentation directly in napari.
    Features parameter selection for training, dynamic loss plotting and automatic saving of the best weights during
    training through validation."""

    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
        data_path="",
        label_path="",
        results_path="",
        model_index=0,
        loss_index=0,
        epochs=10,
        samples=15,
        batch=1,
        val_interval=2,
    ):
        """Creates a Trainer tab widget with the following functionalities :

        * First tab : Dataset parameters
            * A filetype choice to select images in a folder

            * A button to choose the folder containing the images of the dataset. Validation files are chosen automatically from the whole dataset.

            * A button to choose the label folder (must have matching number and name of images)

            * A button to choose where to save the results (weights). Defaults to the plugin's models/saved_weights folder

            * A dropdown menu to choose which model to train

        * Second tab : Training parameters

            * A dropdown menu to choose which loss function to use (see https://docs.monai.io/en/stable/losses.html)

            * A spin box to choose the number of epochs to train for

            * A spin box to choose the batch size during training

            * A spin box to choose the number of samples to take from an image when training

            * A spin box to choose the validation interval

        TODO:

        * Choice of validation proportion, sampling behaviour (toggle), maybe in a data pre-processing tab

        * Custom model loading


        Args:
            viewer: napari viewer to display the widget in

            data_path (str): path to images

            label_path (str): path to labels

            results_path (str): path to results

            model_index (int): model to select by default

            loss_index (int): loss to select by default

            epochs (uint): number of epochs

            samples (uint):  number of samples

            batch (uint): batch size

            val_interval (uint) : epoch interval for validation

        """

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
        self.data_path = directory

        lab_directory = os.path.dirname(os.path.realpath(__file__)) + str(
            Path("/models/dataset/lab_sem")
        )
        self.label_path = lab_directory

        self.images_filepaths = sorted(
            glob.glob(os.path.join(directory, "*.tif"))
        )

        self.labels_filepaths = sorted(
            glob.glob(os.path.join(lab_directory, "*.tif"))
        )

        #######################
        #######################
        #######################
        if results_path == "":
            self.results_path = "C:/Users/Cyril/Desktop/test/models"
        else:
            self.results_path = results_path

        if data_path != "":
            self.data_path = data_path

        if label_path != "":
            self.label_path = label_path

        # recover default values
        self.num_samples = samples
        """Number of samples to extract"""
        self.batch_size = batch

        self.max_epochs = epochs

        self.val_interval = val_interval
        """At which epochs to perform validation. E.g. if 2, will run validation on epochs 2,4,6..."""
        self.model = None  # TODO : custom model loading ?
        self.worker = None
        """Training worker for multithreading"""
        self.data = None

        self.loss_dict = {
            "Dice loss": DiceLoss(sigmoid=True),
            "Focal loss": FocalLoss(),
            "Dice-Focal loss": DiceFocalLoss(sigmoid=True, lambda_dice=0.5),
            "Generalized Dice loss": GeneralizedDiceLoss(sigmoid=True),
            "DiceCELoss": DiceCELoss(sigmoid=True),
            "Tversky loss": TverskyLoss(sigmoid=True),
        }
        """Dict of loss functions"""

        self.canvas = None
        """Canvas to plot loss and dice metric in"""
        self.train_loss_plot = None
        """Plot for loss"""
        self.dice_metric_plot = None
        """Plot for dice metric"""
        self.dock_widgets = []
        """Pointer to a dock widget containing the FigureCanvas, used to remove the docking widget with :py:func:`~close`"""

        self.model_choice.setCurrentIndex(model_index)

        ################################
        # interface
        self.epoch_choice = QSpinBox()
        self.epoch_choice.setValue(self.max_epochs)
        self.epoch_choice.setRange(2, 1000)
        # self.epoch_choice.setSingleStep(2)
        self.epoch_choice.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.lbl_epoch_choice = QLabel("Number of epochs : ", self)

        self.loss_choice = QComboBox()
        self.loss_choice.setCurrentIndex(loss_index)
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

        self.val_interval_choice = QSpinBox()
        self.val_interval_choice.setValue(self.val_interval)
        self.val_interval_choice.setRange(1, 10)
        self.val_interval_choice.setSizePolicy(
            QSizePolicy.Fixed, QSizePolicy.Fixed
        )
        self.lbl_val_interv_choice = QLabel("Validation interval : ", self)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        """Dock widget containing the progress bar"""

        self.btn_start = QPushButton("Start training")
        self.btn_start.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn_start.clicked.connect(self.start)

        self.btn_model_path.setVisible(False)
        self.lbl_model_path.setVisible(False)

        self.build()

    def check_ready(self):
        """
        Checks that the paths to the images and labels are correctly set

        Returns:

            * True if paths are set correctly (!=[])

            * False and displays a warning if not

        """
        if self.images_filepaths != [""] and self.labels_filepaths != [""]:
            return True
        else:
            warnings.formatwarning = utils.format_Warning
            warnings.warn("Image and label paths are not correctly set")
            return False

    def build(self):
        """Builds the layout of the widget and creates the following tabs and prompts:

        * Model parameters :

            * Choice of file type for data

            * Dialog for images folder

            * Dialog for label folder

            * Dialog for results folder

            * Model choice

            * Number of samples to extract from images

            * Next tab

            * Close

        * Training parameters :

            * Loss function choice

            * Batch size choice

            * Epochs choice

            * Validation interval choice

            * Previous tab

            * Start (see :py:func:`~start`)"""

        model_tab = QWidget()
        model_tab.setSizePolicy(
            QSizePolicy.MinimumExpanding, QSizePolicy.Fixed
        )
        ###### first tab : model and dataset choices
        model_tab_layout = QVBoxLayout()
        model_tab_layout.setContentsMargins(0, 0, 1, 11)
        model_tab_layout.setSizeConstraint(QLayout.SetFixedSize)

        model_tab_layout.addWidget(
            utils.combine_blocks(self.filetype_choice, self.lbl_filetype),
            alignment=utils.LEFT_AL,
        )  # file extension

        model_tab_layout.addWidget(
            utils.combine_blocks(self.btn_image_files, self.lbl_image_files),
            alignment=utils.LEFT_AL,
        )  # volumes
        if self.data_path != "":
            self.lbl_image_files.setText(self.data_path)

        model_tab_layout.addWidget(
            utils.combine_blocks(self.btn_label_files, self.lbl_label_files),
            alignment=utils.LEFT_AL,
        )  # labels
        if self.label_path != "":
            self.lbl_label_files.setText(self.label_path)

        # model_tab_layout.addWidget( # TODO : add custom model choice
        #     utils.combine_blocks(self.model_choice, self.lbl_model_choice)
        # )  # model choice

        model_tab_layout.addWidget(
            utils.combine_blocks(self.btn_result_path, self.lbl_result_path),
            alignment=utils.LEFT_AL,
        )  # results folder
        if self.results_path != "":
            self.lbl_result_path.setText(self.results_path)

        model_tab_layout.addWidget(
            utils.combine_blocks(self.model_choice, self.lbl_model_choice),
            alignment=utils.LEFT_AL,
        )  # model choice

        model_tab_layout.addWidget(
            self.lbl_sample_choice, alignment=utils.LEFT_AL
        )
        model_tab_layout.addWidget(
            self.sample_choice, alignment=utils.LEFT_AL
        )  # number of samples
        # TODO add transfo tab and add there ?
        utils.add_blank(self, model_tab_layout)

        model_tab_layout.addWidget(
            self.btn_next, alignment=utils.LEFT_AL
        )  # next
        utils.add_blank(self, model_tab_layout)
        model_tab_layout.addWidget(
            self.btn_close, alignment=utils.LEFT_AL
        )  # close

        #####################
        train_tab = QWidget()
        train_tab.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        ####### second tab : training parameters
        train_tab_layout = QVBoxLayout()
        train_tab_layout.setContentsMargins(0, 0, 1, 11)
        train_tab_layout.setSizeConstraint(QLayout.SetFixedSize)

        train_tab_layout.addWidget(
            utils.combine_blocks(self.loss_choice, self.lbl_loss_choice),
            alignment=utils.LEFT_AL,
        )  # loss choice

        spinbox_spacing = 110

        train_tab_layout.addWidget(
            utils.combine_blocks(
                self.batch_choice, self.lbl_batch_choice, spinbox_spacing
            ),
            alignment=utils.LEFT_AL,
        )  # batch size
        train_tab_layout.addWidget(
            utils.combine_blocks(
                self.epoch_choice, self.lbl_epoch_choice, spinbox_spacing
            ),
            alignment=utils.LEFT_AL,
        )  # epochs
        train_tab_layout.addWidget(
            utils.combine_blocks(
                self.val_interval_choice,
                self.lbl_val_interv_choice,
                spinbox_spacing,
            ),
            alignment=utils.LEFT_AL,
        )  # validation interval

        utils.add_blank(self, train_tab_layout)

        train_tab_layout.addWidget(
            self.btn_prev, alignment=utils.LEFT_AL
        )  # previous
        utils.add_blank(self, train_tab_layout)
        train_tab_layout.addWidget(
            self.btn_start, alignment=utils.LEFT_AL
        )  # start

        self.setSizePolicy(
            QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding
        )

        utils.make_scrollable(
            contained_layout=model_tab_layout, containing_widget=model_tab
        )  # , min_wh=[190, 100], max_wh=[200,1000])
        self.addTab(model_tab, "Model parameters")

        utils.make_scrollable(
            contained_layout=train_tab_layout,
            containing_widget=train_tab,
            min_wh=[250, 100],
        )
        self.addTab(train_tab, "Training parameters")

    def show_dialog_lab(self):
        """Shows the  dialog to load label files in a path, loads them (see :doc:model_framework) and changes the widget
        label :py:attr:`self.lbl_label` accordingly"""
        f_name = utils.open_file_dialog(self, self._default_path)

        if f_name:
            self.label_path = f_name
            self.lbl_label.setText(self.label_path)

    def show_dialog_dat(self):
        """Shows the  dialog to load images files in a path, loads them (see :doc:model_framework) and changes the
        widget label :py:attr:`self.lbl_dat` accordingly"""
        f_name = utils.open_file_dialog(self, self._default_path)

        if f_name:
            self.data_path = f_name
            self.lbl_dat.setText(self.data_path)

    def start(self):
        """
        Initiates the :py:func:`train` function as a worker and does the following :

        * Checks that filepaths are set correctly using :py:func:`check_ready`

        * If self.worker is None : creates a worker and starts the training

        * If the button is clicked while training, stops the model once it finishes the next validation step and saves the results if better

        * When the worker yields after a validation step, plots the loss if epoch >= validation_step (to avoid empty plot on first validation)

        * When the worker finishes, clears the memory (tries to for now)

        TODO:

        * Fix memory leak


        Returns: Returns empty immediately if the file paths are not set correctly.

        """
        self.print_and_log("Starting...")
        self.print_and_log("*" * 20)

        if not self.check_ready():  # issues a warning if not ready
            err = "Aborting, please set all required paths"
            self.print_and_log(err)
            raise ValueError(err)
            return

        if self.worker is not None:
            if self.worker.is_running:
                pass
            else:
                self.worker.start()
                self.btn_start.setText("Running... Click to stop")
        else:

            self.num_samples = self.sample_choice.value()
            self.batch_size = self.batch_choice.value()
            self.val_interval = self.val_interval_choice.value()
            self.data = self.create_train_dataset_dict()
            self.max_epochs = self.epoch_choice.value()

            model_dict = {
                "class": self.get_model(self.model_choice.currentText()),
                "name": self.model_choice.currentText(),
            }

            self.worker = self.train(
                device=self.get_device(),
                model_dict=model_dict,
                data_dicts=self.data,
                max_epochs=self.max_epochs,
                loss_function=self.get_loss(self.loss_choice.currentText()),
                val_interval=self.val_interval,
                batch_size=self.batch_size,
                results_path=self.results_path,
                num_samples=self.num_samples,
                logger=lambda text: self.worker_print_and_log(self, text),
            )

            self.worker.start()
            self.btn_close.setVisible(False)

            self.worker.started.connect(self.on_start)

            self.worker.yielded.connect(
                lambda data: self.on_yield(data, widget=self)
            )
            self.worker.finished.connect(self.on_finish)

            self.worker.errored.connect(self.on_error)

        if self.worker.is_running:
            self.print_and_log(
                f"Stop requested at {utils.get_time()}. \nWaiting for next validation step..."
            )
            self.btn_start.setText("Stopping... Please wait for next saving")
            self.worker.quit()
        else:
            # self.worker.start()
            self.btn_start.setText("Running...  Click to stop")

    def on_start(self):

        self.display_status_report()

        self.print_and_log(f"Worker started at {utils.get_time()}")
        self.print_and_log("\nWorker is running...")
        self.print_and_log(f"Saving results to : {self.results_path}")

    def on_finish(self):
        self.print_and_log(f"\nWorker finished at {utils.get_time()}")

        self.print_and_log(f"Saving last loss plot at {self.results_path}")
        if self.canvas is not None:
            self.canvas.figure.savefig(
                (
                    self.results_path
                    + f"/final_metric_plots_{utils.get_date_time()}.png"
                ),
                format="png",
            )
        self.print_and_log("Done")
        self.print_and_log("*" * 10)

        self.btn_start.setText("Start")
        self.btn_close.setVisible(True)
        self.clean_cache()

    def on_error(self):
        self.print_and_log(f"WORKER ERRORED at {utils.get_time()}")
        self.clean_cache()

    @staticmethod
    def on_yield(data, widget):
        # print(
        #     f"\nCatching results : for epoch {data['epoch']}, loss is {data['losses']} and validation is {data['val_metrics']}"
        # )
        widget.progress.setValue(
            100 * (data["epoch"] + 1) // widget.max_epochs
        )
        widget.update_loss_plot(data["losses"], data["val_metrics"])

    def clean_cache(self):
        """Attempts to clear memory after training"""
        # del self.worker
        self.worker = None
        # if self.model is not None:
        #     del self.model
        #     self.model = None

        # del self.data
        # self.close()
        # del self
        if self.get_device(show=False).type == "cuda":
            self.empty_cuda_cache()

    def plot_loss(self, loss, dice_metric):
        """Creates two subplots to plot the training loss and validation metric"""
        with plt.style.context("dark_background"):
            # update loss
            self.train_loss_plot.set_title("Epoch average loss")
            self.train_loss_plot.set_xlabel("Epoch")
            self.train_loss_plot.set_ylabel("Loss")
            x = [i + 1 for i in range(len(loss))]
            y = loss
            self.train_loss_plot.plot(x, y)
            self.train_loss_plot.set_ylim(0, 1)

            # update metrics
            x = [self.val_interval * (i + 1) for i in range(len(dice_metric))]
            y = dice_metric

            epoch_min = (np.argmax(y) + 1) * self.val_interval
            dice_min = np.max(y)

            self.dice_metric_plot.plot(x, y, zorder=1)
            self.dice_metric_plot.set_ylim(0, 1)
            self.dice_metric_plot.set_title(
                "Validation metric : Mean Dice coefficient"
            )
            self.dice_metric_plot.set_xlabel("Epoch")
            self.dice_metric_plot.set_ylabel("Dice")

            self.dice_metric_plot.scatter(
                epoch_min,
                dice_min,
                c="r",
                label="Maximum Dice coeff.",
                zorder=5,
            )
            self.dice_metric_plot.legend(
                facecolor="#262930", loc="lower right"
            )
            self.canvas.draw_idle()

            plot_path = self.results_path + "/Loss_plots"
            os.makedirs(plot_path, exist_ok=True)
            if self.canvas is not None:
                self.canvas.figure.savefig(
                    (
                        plot_path
                        + f"/checkpoint_metric_plots_{utils.get_date_time()}.png"
                    ),
                    format="png",
                )

    def update_loss_plot(self, loss, metric):
        """
        Updates the plots on subsequent validation steps.
        Creates the plot on the second validation step (epoch == val_interval*2).
        Updates the plot on subsequent validation steps.
        Epoch is obtained from the length of the loss vector.

        Returns: returns empty if the epoch is < than 2 * validation interval.
        """

        epoch = len(loss)
        if epoch < self.val_interval * 2:
            return
        elif epoch == self.val_interval * 2:
            bckgrd_color = (0, 0, 0, 0)  # '#262930'
            with plt.style.context("dark_background"):

                self.canvas = FigureCanvas(Figure(figsize=(10, 1.5)))
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
                    bottom=0.3,
                    right=0.95,
                    top=0.8,
                    wspace=0.2,
                    hspace=0,
                )

            self.canvas.setSizePolicy(
                QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding
            )

            # tab_index = self.addTab(self.canvas, "Loss plot")
            # self.setCurrentIndex(tab_index)
            plot_dock = self._viewer.window.add_dock_widget(
                self.canvas, name="Loss plots", area="bottom"
            )
            self.dock_widgets.append(plot_dock)
            self.plot_loss(loss, metric)
        else:
            with plt.style.context("dark_background"):

                self.train_loss_plot.cla()
                self.dice_metric_plot.cla()

                self.plot_loss(loss, metric)

    @staticmethod
    @thread_worker
    def train(
        device,
        model_dict,
        data_dicts,
        max_epochs,
        loss_function,
        val_interval,
        batch_size,
        results_path,
        num_samples,
        logger,
    ):  # TODO : turn into static
        """Trains the Pytorch model for num_epochs, with the selected model and data, using the chosen batch size,
        validation interval, loss function, and number of samples."""

        model_name = model_dict["name"]
        model_class = model_dict["class"]
        model = model_class.get_net()
        model = model.to(device)

        epoch_loss_values = []
        val_metric_values = []

        # TODO param : % of validation from training set
        train_files, val_files = (
            data_dicts[0 : int(len(data_dicts) * 0.9)],
            data_dicts[int(len(data_dicts) * 0.9) :],
        )
        # print("train/val")
        # print(train_files)
        # print(val_files)
        # TODO : param stretch factor if anisotropic ?
        # TODO : param ROI size
        sample_loader = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                RandSpatialCropSamplesd(
                    keys=["image", "label"],
                    roi_size=(
                        110,
                        110,
                        110,
                    ),  # TODO multiply by axis_stretch_factor
                    max_roi_size=(120, 120, 120),
                    num_samples=num_samples,
                ),
                Orientationd(keys=["image", "label"], axcodes="PLI"),
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
                RandFlipd(keys=["image", "label"]),
                RandRotate90d(keys=["image", "label"]),
                RandAffined(
                    keys=["image", "label"],
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
        post_pred = EnsureType()  # AsDiscrete(threshold=0.3)
        post_label = EnsureType()

        optimizer = torch.optim.Adam(model.parameters(), 1e-3)
        dice_metric = DiceMetric(include_background=True, reduction="mean")

        best_metric = -1
        best_metric_epoch = -1

        time = utils.get_date_time()

        weights_filename = f"{model_name}_best_metric" + f"_{time}.pth"
        if device.type == "cuda":
            logger("\nUsing GPU :")
            logger(torch.cuda.get_device_name(0))
        else:
            logger("Using CPU")

        for epoch in range(max_epochs):
            logger("-" * 10)
            logger(f"Epoch {epoch + 1}/{max_epochs}")
            if device.type == "cuda":
                logger("Memory Usage:")
                alloc_mem = round(
                    torch.cuda.memory_allocated(0) / 1024**3, 1
                )
                reserved_mem = round(
                    torch.cuda.memory_reserved(0) / 1024**3, 1
                )
                logger(f"Allocated: {alloc_mem}GB")
                logger(f"Cached: {reserved_mem}GB")

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
                outputs = model_class.get_output(  # AsDiscrete(threshold=0.7)(
                    model, inputs
                )
                print(f"OUT : {outputs.shape}")
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().item()
                logger(
                    f"* {step}/{len(train_ds) // train_loader.batch_size}, "
                    f"Train_loss: {loss.detach().item():.4f}"
                )
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            logger(f"-> Epoch: {epoch + 1}, Average loss: {epoch_loss:.4f}")

            if (epoch + 1) % val_interval == 0:
                model.eval()
                with torch.no_grad():
                    for val_data in val_loader:
                        val_inputs, val_labels = (
                            val_data["image"].to(device),
                            val_data["label"].to(device),
                        )

                        val_outputs = model_class.get_validation(
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

                        # print(len(val_outputs))
                        # print(len(val_labels))

                        dice_metric(y_pred=val_outputs, y=val_labels)

                    metric = dice_metric.aggregate().detach().item()
                    dice_metric.reset()

                    val_metric_values.append(metric)

                    train_report = {
                        "epoch": epoch,
                        "losses": epoch_loss_values,
                        "val_metrics": val_metric_values,
                    }
                    yield train_report

                    if metric > best_metric:
                        best_metric = metric
                        best_metric_epoch = epoch + 1
                        torch.save(
                            model.state_dict(),
                            os.path.join(results_path, weights_filename),
                        )
                        logger("Saved best metric model")
                    logger(
                        f"> Current epoch: {epoch + 1}, Current mean dice: {metric:.4f}"
                        f"\nBest mean dice: {best_metric:.4f} "
                        f"at epoch: {best_metric_epoch}"
                    )
        logger("=" * 10)
        logger(
            f"Train completed, best_metric: {best_metric:.4f} "
            f"at epoch: {best_metric_epoch}"
        )
        # del device
        # del model_id
        # del model_name
        # del model
        # del data_dicts
        # del max_epochs
        # del loss_function
        # del val_interval
        # del batch_size
        # del results_path
        # del num_samples
        # del best_metric
        # del best_metric_epoch

        # self.close()
