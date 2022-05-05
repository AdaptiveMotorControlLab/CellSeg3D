import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import napari
import numpy as np
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.figure import Figure

# MONAI
from monai.losses import DiceCELoss
from monai.losses import DiceFocalLoss
from monai.losses import DiceLoss
from monai.losses import FocalLoss
from monai.losses import GeneralizedDiceLoss
from monai.losses import TverskyLoss

# Qt
from qtpy.QtWidgets import QLabel
from qtpy.QtWidgets import QProgressBar
from qtpy.QtWidgets import QSizePolicy

# local
from napari_cellseg_annotator import interface as ui
from napari_cellseg_annotator import utils
from napari_cellseg_annotator.model_framework import ModelFramework
from napari_cellseg_annotator.model_workers import TrainingWorker

NUMBER_TABS = 3


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
        epochs=5,
        samples=2,
        batch=1,
        val_interval=1,
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

        self.data_path = ""
        self.label_path = ""
        self.results_path = ""
        ######################
        ######################
        ######################
        # TEST TODO REMOVE
        import glob

        if utils.ENABLE_TEST_MODE():
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

            if results_path == "":
                self.results_path = "C:/Users/Cyril/Desktop/test/models"
            else:
                self.results_path = results_path

            if data_path != "":
                self.data_path = data_path

            if label_path != "":
                self.label_path = label_path
        #######################
        #######################
        #######################

        # recover default values
        self.num_samples = samples
        """Number of samples to extract"""
        self.batch_size = batch
        """Batch size"""
        self.max_epochs = epochs
        """Epochs"""
        self.val_interval = val_interval
        """At which epochs to perform validation. E.g. if 2, will run validation on epochs 2,4,6..."""
        self.patch_size = []
        """The size of samples to be extracted from images"""

        self.model = None  # TODO : custom model loading ?
        self.worker = None
        """Training worker for multithreading, should be a TrainingWorker instance from :doc:model_workers.py"""
        self.data = None
        """Data dictionary containing file paths"""

        self.loss_dict = {
            "Dice loss": DiceLoss(sigmoid=True),
            "Focal loss": FocalLoss(),
            "Dice-Focal loss": DiceFocalLoss(sigmoid=True, lambda_dice=0.5),
            "Generalized Dice loss": GeneralizedDiceLoss(sigmoid=True),
            "DiceCELoss": DiceCELoss(sigmoid=True, lambda_ce=0.5),
            "Tversky loss": TverskyLoss(sigmoid=True),
        }
        """Dict of loss functions"""

        self.canvas = None
        """Canvas to plot loss and dice metric in"""
        self.train_loss_plot = None
        """Plot for loss"""
        self.dice_metric_plot = None
        """Plot for dice metric"""
        self.plot_dock = None
        """Docked widget with plots"""

        self.model_choice.setCurrentIndex(model_index)

        ################################
        # interface
        self.epoch_choice = ui.make_n_spinboxes(
            min=2, max=1000, default=self.max_epochs
        )
        self.lbl_epoch_choice = QLabel("Number of epochs : ", self)

        self.loss_choice, self.lbl_loss_choice = ui.make_combobox(
            sorted(self.loss_dict.keys()), label="Loss function"
        )
        self.loss_choice.setCurrentIndex(loss_index)

        self.sample_choice = ui.make_n_spinboxes(
            min=2, max=50, default=self.num_samples
        )
        self.lbl_sample_choice = QLabel("Number of patches per image : ", self)
        self.sample_choice.setVisible(False)
        self.lbl_sample_choice.setVisible(False)

        self.batch_choice = ui.make_n_spinboxes(
            min=1, max=10, default=self.batch_size
        )
        self.lbl_batch_choice = QLabel("Batch size : ", self)

        self.val_interval_choice = ui.make_n_spinboxes(
            default=self.val_interval
        )
        self.lbl_val_interv_choice = QLabel("Validation interval : ", self)

        self.augment_choice = ui.make_checkbox("Augment data")

        # TODO add self.tabs, self.close_buttons etc...
        self.close_buttons = [
            self.make_close_button() for i in range(NUMBER_TABS)
        ]
        """Close buttons list for each tab"""

        self.patch_size_widgets = ui.make_n_spinboxes(3, 10, 1023, 120)

        self.patch_size_lbl = [
            QLabel(f"Size of patch in {axis} :") for axis in "xyz"
        ]
        for w in self.patch_size_widgets:
            w.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            w.setVisible(False)
        for l in self.patch_size_lbl:
            l.setVisible(False)
        self.sampling_container = QLabel()

        self.patch_choice = ui.make_checkbox(
            "Extract patches from images", func=self.toggle_patch_dims
        )
        self.patch_choice.clicked.connect(self.toggle_patch_dims)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        """Dock widget containing the progress bar"""

        self.btn_start = ui.make_button("Start training", self.start)

        self.btn_model_path.setVisible(False)
        self.lbl_model_path.setVisible(False)

        self.build()

    def toggle_patch_dims(self):
        if self.patch_choice.isChecked():
            [w.setVisible(True) for w in self.patch_size_widgets]
            [l.setVisible(True) for l in self.patch_size_lbl]
            self.sample_choice.setVisible(True)
            self.lbl_sample_choice.setVisible(True)
            self.sampling_container.setVisible(True)
        else:
            [w.setVisible(False) for w in self.patch_size_widgets]
            [l.setVisible(False) for l in self.patch_size_lbl]
            self.sample_choice.setVisible(False)
            self.lbl_sample_choice.setVisible(False)
            self.sampling_container.setVisible(False)

    def check_ready(self):
        """
        Checks that the paths to the images and labels are correctly set

        Returns:

            * True if paths are set correctly (!=[""])

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

        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.MinimumExpanding)
        ########
        ################
        ########################
        # first tab : model and dataset choices
        data_tab, data_tab_layout = ui.make_container_widget()
        ################
        # first group : Data
        data_group, data_layout = ui.make_group("Data")

        data_layout.addWidget(
            ui.combine_blocks(self.filetype_choice, self.lbl_filetype),
            alignment=ui.LEFT_AL,
        )  # file extension

        data_layout.addWidget(
            ui.combine_blocks(self.btn_image_files, self.lbl_image_files),
            alignment=ui.LEFT_AL,
        )  # volumes
        if self.data_path != "":
            self.lbl_image_files.setText(self.data_path)

        data_layout.addWidget(
            ui.combine_blocks(self.btn_label_files, self.lbl_label_files),
            alignment=ui.LEFT_AL,
        )  # labels
        if self.label_path != "":
            self.lbl_label_files.setText(self.label_path)

        # data_tab_layout.addWidget( # TODO : add custom model choice
        #     ui.combine_blocks(self.model_choice, self.lbl_model_choice)
        # )  # model choice

        data_layout.addWidget(
            ui.combine_blocks(self.btn_result_path, self.lbl_result_path),
            alignment=ui.LEFT_AL,
        )  # results folder
        if self.results_path != "":
            self.lbl_result_path.setText(self.results_path)

        data_group.setLayout(data_layout)
        data_tab_layout.addWidget(data_group, alignment=ui.LEFT_AL)
        # end of first group : Data
        ui.add_blank(widget=data_tab, layout=data_tab_layout)
        ################
        ui.add_blank(self, data_tab_layout)
        ################
        # buttons

        data_tab_layout.addWidget(
            self.make_next_button(), alignment=ui.LEFT_AL
        )  # next
        ui.add_blank(self, data_tab_layout)
        data_tab_layout.addWidget(
            self.close_buttons[0], alignment=ui.LEFT_AL
        )  # close

        ##################
        ############
        ######
        # second tab : image sizes, data augmentation, patches size and behaviour
        ######
        ############
        ##################
        augment_tab_w, augment_tab_l = ui.make_container_widget()
        ##################
        sampling_group_w, sampling_group_l = ui.make_group("Sampling")

        sampling_group_l.addWidget(
            self.patch_choice, alignment=ui.LEFT_AL
        )  # extract patches or not

        #######################################################
        patch_size_w, patch_size_l = ui.make_container_widget()
        [
            patch_size_l.addWidget(widget, alignment=ui.LEFT_AL)
            for widgts in zip(self.patch_size_lbl, self.patch_size_widgets)
            for widget in widgts
        ]  # patch sizes

        patch_size_w.setLayout(patch_size_l)
        #######################################################
        #######################################################
        sampling_w, sampling_l = ui.make_container_widget()

        sampling_l.addWidget(self.lbl_sample_choice, alignment=ui.LEFT_AL)
        sampling_l.addWidget(
            self.sample_choice, alignment=ui.LEFT_AL
        )  # number of samples

        sampling_w.setLayout(sampling_l)
        #######################################################
        self.sampling_container = ui.combine_blocks(
            sampling_w, patch_size_w, horizontal=False, min_spacing=130, b=5
        )
        self.sampling_container.setVisible(False)
        #######################################################
        sampling_group_l.addWidget(self.sampling_container)
        sampling_group_w.setLayout(sampling_group_l)
        augment_tab_l.addWidget(sampling_group_w)
        #######################
        ui.add_blank(augment_tab_w, augment_tab_l)
        #######################
        augment_group_w, augment_group_l = ui.make_group("Augmentation")
        augment_group_l.addWidget(
            self.augment_choice, alignment=ui.LEFT_AL
        )  # augment data toggle
        self.augment_choice.toggle()

        augment_group_w.setLayout(augment_group_l)
        augment_tab_l.addWidget(augment_group_w)
        #######################
        ui.add_blank(augment_tab_w, augment_tab_l)
        #######################

        augment_tab_l.addWidget(
            ui.combine_blocks(
                first=self.make_prev_button(),
                second=self.make_next_button(),
                l=1,
            ),
            alignment=ui.LEFT_AL,
        )

        augment_tab_l.addWidget(self.close_buttons[1], alignment=ui.LEFT_AL)
        ##################
        ############
        ######
        # third tab : training parameters
        ######
        ############
        ##################
        train_tab, train_tab_layout = ui.make_container_widget()
        ##################
        # solo groups for loss and model
        ui.add_blank(train_tab, train_tab_layout)

        ui.make_group(
            "Model",
            solo_dict={
                "widget": self.model_choice,
                "layout": train_tab_layout,
            },
        )  # model choice
        self.lbl_model_choice.setVisible(False)

        ui.add_blank(train_tab, train_tab_layout)

        ui.make_group(
            "Loss",
            solo_dict={"widget": self.loss_choice, "layout": train_tab_layout},
        )  # loss choice
        self.lbl_loss_choice.setVisible(False)

        # end of solo groups for loss and model
        ##################
        ui.add_blank(train_tab, train_tab_layout)
        ##################
        # training params group

        train_param_group_w, train_param_group_l = ui.make_group(
            "Training parameters", R=1, B=5, T=11
        )

        spacing = 20
        train_param_group_l.addWidget(
            ui.combine_blocks(
                self.batch_choice,
                self.lbl_batch_choice,
                min_spacing=spacing,
                horizontal=False,
                l=5,
                t=5,
                r=5,
                b=5,
            ),
            alignment=ui.LEFT_AL,
        )  # batch size
        train_param_group_l.addWidget(
            ui.combine_blocks(
                self.epoch_choice,
                self.lbl_epoch_choice,
                min_spacing=spacing,
                horizontal=False,
                l=5,
                t=5,
                r=5,
                b=5,
            ),
            alignment=ui.LEFT_AL,
        )  # epochs
        train_param_group_l.addWidget(
            ui.combine_blocks(
                self.val_interval_choice,
                self.lbl_val_interv_choice,
                min_spacing=spacing,
                horizontal=False,
                l=5,
                t=5,
                r=5,
                b=5,
            ),
            alignment=ui.LEFT_AL,
        )  # validation interval

        train_param_group_w.setLayout(train_param_group_l)
        train_tab_layout.addWidget(train_param_group_w)
        # end of training params group
        ##################
        # buttons

        ui.add_blank(self, train_tab_layout)

        train_tab_layout.addWidget(
            self.make_prev_button(), alignment=ui.LEFT_AL
        )  # previous

        train_tab_layout.addWidget(
            self.btn_start, alignment=ui.LEFT_AL
        )  # start
        ui.add_blank(self, train_tab_layout)

        train_tab_layout.addWidget(
            self.close_buttons[2],
            alignment=ui.LEFT_AL,
        )
        ##################
        ############
        ######
        # end of tab layouts

        ui.make_scrollable(
            contained_layout=data_tab_layout,
            containing_widget=data_tab,
            min_wh=[100, 150],
        )  # , max_wh=[200,1000])
        self.addTab(data_tab, "Data")

        ui.make_scrollable(
            contained_layout=augment_tab_l,
            containing_widget=augment_tab_w,
            min_wh=[100, 200],
        )
        self.addTab(augment_tab_w, "Augmentation")

        ui.make_scrollable(
            contained_layout=train_tab_layout,
            containing_widget=train_tab,
            min_wh=[100, 200],
        )
        self.addTab(train_tab, "Training")

    def show_dialog_lab(self):
        """Shows the  dialog to load label files in a path, loads them (see :doc:model_framework) and changes the widget
        label :py:attr:`self.lbl_label` accordingly"""
        f_name = ui.open_file_dialog(self, self._default_path)

        if f_name:
            self.label_path = f_name
            self.lbl_label.setText(self.label_path)

    def show_dialog_dat(self):
        """Shows the  dialog to load images files in a path, loads them (see :doc:model_framework) and changes the
        widget label :py:attr:`self.lbl_dat` accordingly"""
        f_name = ui.open_file_dialog(self, self._default_path)

        if f_name:
            self.data_path = f_name
            self.lbl_dat.setText(self.data_path)

    def send_log(self, text):
        self.log.print_and_log(text)

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

        if not self.check_ready():  # issues a warning if not ready
            err = "Aborting, please set all required paths"
            self.log.print_and_log(err)
            warnings.warn(err)
            return

        if self.worker is not None:
            if self.worker.is_running:
                pass
            else:
                self.worker.start()
                self.btn_start.setText("Running... Click to stop")
        else:  # starting a new job goes here
            self.log.print_and_log("Starting...")
            self.log.print_and_log("*" * 20)

            self.reset_loss_plot()

            self.num_samples = self.sample_choice.value()
            self.batch_size = self.batch_choice.value()
            self.val_interval = self.val_interval_choice.value()
            self.data = self.create_train_dataset_dict()
            self.max_epochs = self.epoch_choice.value()

            self.patch_size = []
            [
                self.patch_size.append(w.value())
                for w in self.patch_size_widgets
            ]

            model_dict = {
                "class": self.get_model(self.model_choice.currentText()),
                "name": self.model_choice.currentText(),
            }

            self.results_path = (
                self.results_path
                + f"/{model_dict['name']}_results_{utils.get_date_time()}"
            )

            os.makedirs(
                self.results_path, exist_ok=False
            )  # avoid overwrite where possible

            self.log.print_and_log(
                f"Notice : Saving results to : {self.results_path}"
            )

            self.worker = TrainingWorker(
                device=self.get_device(),
                model_dict=model_dict,
                data_dicts=self.data,
                max_epochs=self.max_epochs,
                loss_function=self.get_loss(self.loss_choice.currentText()),
                val_interval=self.val_interval,
                batch_size=self.batch_size,
                results_path=self.results_path,
                sampling=self.patch_choice.isChecked(),
                num_samples=self.num_samples,
                sample_size=self.patch_size,
                do_augmentation=self.augment_choice.isChecked(),
            )

            [btn.setVisible(False) for btn in self.close_buttons]

            self.worker.log_signal.connect(self.log.print_and_log)

            self.worker.started.connect(self.on_start)

            self.worker.yielded.connect(
                lambda data: self.on_yield(data, widget=self)
            )
            self.worker.finished.connect(self.on_finish)

            self.worker.errored.connect(self.on_error)

        if self.worker.is_running:
            self.log.print_and_log(
                f"Stop requested at {utils.get_time()}. \nWaiting for next validation step..."
            )
            self.btn_start.setText("Stopping... Please wait for next saving")
            self.worker.quit()
        else:
            self.worker.start()
            self.btn_start.setText("Running...  Click to stop")

    def on_start(self):
        """Catches started signal from worker"""
        if self.plot_dock is not None:
            self._viewer.window.remove_dock_widget(self.plot_dock)
            self.plot_dock = None

        self.display_status_report()

        self.log.print_and_log(f"Worker started at {utils.get_time()}")
        self.log.print_and_log("\nWorker is running...")

    def on_finish(self):
        """Catches finished signal from worker"""
        self.log.print_and_log(f"\nWorker finished at {utils.get_time()}")

        self.log.print_and_log(f"Saving last loss plot at {self.results_path}")

        if self.canvas is not None:
            self.canvas.figure.savefig(
                (
                    self.results_path
                    + f"/final_metric_plots_{utils.get_date_time()}.png"
                ),
                format="png",
            )

        self.log.print_and_log("Auto-saving log")
        self.save_log()

        self.log.print_and_log("Done")
        self.log.print_and_log("*" * 10)

        self.btn_start.setText("Start")
        [btn.setVisible(True) for btn in self.close_buttons]

        self.worker = None
        self.empty_cuda_cache()
        # self.clean_cache() # trying to fix memory leak

    def on_error(self):
        """Catches errored signal from worker"""
        self.log.print_and_log(f"WORKER ERRORED at {utils.get_time()}")
        self.empty_cuda_cache()
        # self.clean_cache()

    @staticmethod
    def on_yield(data, widget):
        # print(
        #     f"\nCatching results : for epoch {data['epoch']}, loss is {data['losses']} and validation is {data['val_metrics']}"
        # )
        widget.progress.setValue(
            100 * (data["epoch"] + 1) // widget.max_epochs
        )
        widget.update_loss_plot(data["losses"], data["val_metrics"])

    # def clean_cache(self):
    #     """Attempts to clear memory after training"""
    #     # del self.worker
    #     self.worker = None
    #     # if self.model is not None:
    #     #     del self.model
    #     #     self.model = None
    #
    #     # del self.data
    #     # self.close()
    #     # del self
    #     if self.get_device(show=False).type == "cuda":
    #         self.empty_cuda_cache()

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
            # self.train_loss_plot.set_ylim(0, 1)

            # update metrics
            x = [self.val_interval * (i + 1) for i in range(len(dice_metric))]
            y = dice_metric

            epoch_min = (np.argmax(y) + 1) * self.val_interval
            dice_min = np.max(y)

            self.dice_metric_plot.plot(x, y, zorder=1)
            # self.dice_metric_plot.set_ylim(0, 1)
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
                facecolor=ui.napari_grey, loc="lower right"
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
            self.plot_dock = self._viewer.window.add_dock_widget(
                self.canvas, name="Loss plots", area="bottom"
            )
            self.docked_widgets.append(self.plot_dock)
            self.plot_loss(loss, metric)
        else:
            with plt.style.context("dark_background"):

                self.train_loss_plot.cla()
                self.dice_metric_plot.cla()

                self.plot_loss(loss, metric)

    def reset_loss_plot(self):
        if (
            self.train_loss_plot is not None
            and self.dice_metric_plot is not None
        ):
            with plt.style.context("dark_background"):
                self.train_loss_plot.cla()
                self.dice_metric_plot.cla()
