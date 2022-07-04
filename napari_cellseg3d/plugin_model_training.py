import os
import shutil
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import napari
import numpy as np
import pandas as pd
import torch
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
from qtpy.QtWidgets import QSizePolicy

# local
from napari_cellseg3d import interface as ui
from napari_cellseg3d import utils
from napari_cellseg3d.model_framework import ModelFramework
from napari_cellseg3d.model_workers import TrainingWorker

NUMBER_TABS = 3
DEFAULT_PATCH_SIZE = 64


class Trainer(ModelFramework):
    """A plugin to train pre-defined PyTorch models for one-channel segmentation directly in napari.
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
            * A choice for the file extension of images to be loaded

            * A button to select images folder. Validation files are chosen automatically from the whole dataset.

            * A button to choose the label folder (must have matching number and name regarding images folder)

            * A button to choose where to save the results (weights, log, plots). Defaults to the plugin's models/saved_weights folder

            * A choice of whether to use pre-trained weights or load custom weights if desired

            * A choice of the proportion of the dataset to use for validation.

        * Second tab : Data augmentation

            * A choice for using images as is or extracting smaller patches randomly, with a size and number choice.

            * A toggle for data augmentation (elastic deforms, intensity shift, flipping, etc)

        * Third tab : Training parameters

            * A choice of model to use (see training module guide table)

            * A dropdown menu to choose which loss function to use (see the training module guide table)

            * A spin box to choose the number of epochs to train for

            * A spin box to choose the batch size during training

            * A choice of learning rate for the optimizer

            * A spin box to choose the validation interval

            * A choice of using random or deterministic training

        TODO training plugin:


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
        self.results_path_folder = ""
        """Path to the folder inside the results path that contains all results"""

        self.save_as_zip = False
        """Whether to zip results folder once done. Creates a zipped copy of the results folder."""

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
        self.learning_rate = 1e-3

        self.model = None  # TODO : custom model loading ?
        self.worker = None
        """Training worker for multithreading, should be a TrainingWorker instance from :doc:model_workers.py"""
        self.data = None
        """Data dictionary containing file paths"""
        self.stop_requested = False
        """Whether the worker should stop or not"""
        self.start_time = ""

        self.loss_dict = {
            "Dice loss": DiceLoss(sigmoid=True),
            "Focal loss": FocalLoss(),
            # "BCELoss":nn.BCELoss(),
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

        self.df = None
        self.loss_values = []
        self.validation_values = []

        self.model_choice.setCurrentIndex(model_index)

        ################################
        # interface

        self.zip_choice = ui.make_checkbox("Compress results")

        self.validation_percent_choice = ui.IntIncrementCounter(
            10, 90, default=80, step=1, parent=self
        )

        self.epoch_choice = ui.IntIncrementCounter(
            min=2, max=1000, default=self.max_epochs
        )
        self.lbl_epoch_choice = ui.make_label("Number of epochs : ", self)

        self.loss_choice = ui.DropdownMenu(
            sorted(self.loss_dict.keys()), label="Loss function"
        )
        self.lbl_loss_choice = self.loss_choice.label
        self.loss_choice.setCurrentIndex(loss_index)

        self.sample_choice = ui.IntIncrementCounter(
            min=2, max=50, default=self.num_samples
        )
        self.lbl_sample_choice = ui.make_label(
            "Number of patches per image : ", self
        )
        self.sample_choice.setVisible(False)
        self.lbl_sample_choice.setVisible(False)

        self.batch_choice = ui.IntIncrementCounter(
            min=1, max=10, default=self.batch_size
        )
        self.lbl_batch_choice = ui.make_label("Batch size : ", self)

        self.val_interval_choice = ui.IntIncrementCounter(
            default=self.val_interval
        )
        self.lbl_val_interv_choice = ui.make_label(
            "Validation interval : ", self
        )

        learning_rate_vals = [
            "1e-2",
            "1e-3",
            "1e-4",
            "1e-5",
            "1e-6",
        ]

        self.learning_rate_choice = ui.DropdownMenu(
            learning_rate_vals, label="Learning rate"
        )
        self.lbl_learning_rate_choice = self.learning_rate_choice.label

        self.learning_rate_choice.setCurrentIndex(1)

        self.augment_choice = ui.make_checkbox("Augment data")

        self.close_buttons = [
            self.make_close_button() for i in range(NUMBER_TABS)
        ]
        """Close buttons list for each tab"""

        self.patch_size_widgets = ui.IntIncrementCounter.make_n(
            3, 10, 1024, DEFAULT_PATCH_SIZE
        )

        self.patch_size_lbl = [
            ui.make_label(f"Size of patch in {axis} :", self) for axis in "xyz"
        ]
        for w in self.patch_size_widgets:
            w.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            w.setVisible(False)
        for l in self.patch_size_lbl:
            l.setVisible(False)
        self.sampling_container, l = ui.make_container()

        self.patch_choice = ui.make_checkbox(
            "Extract patches from images", func=self.toggle_patch_dims
        )
        self.patch_choice.clicked.connect(self.toggle_patch_dims)

        self.use_transfer_choice = ui.make_checkbox(
            "Transfer weights", self.toggle_transfer_param
        )

        self.use_deterministic_choice = ui.make_checkbox(
            "Deterministic training", func=self.toggle_deterministic_param
        )
        self.box_seed = ui.IntIncrementCounter(max=10000000, default=23498)
        self.lbl_seed = ui.make_label("Seed", self)
        self.container_seed = ui.combine_blocks(
            self.box_seed, self.lbl_seed, horizontal=False
        )

        self.progress.setVisible(False)
        """Dock widget containing the progress bar"""

        self.btn_start = ui.Button("Start training", self.start)

        self.btn_model_path.setVisible(False)
        self.lbl_model_path.setVisible(False)

        ############################
        ############################
        # tooltips
        self.zip_choice.setToolTip(
            "Checking this will save a copy of the results as a zip folder"
        )
        self.validation_percent_choice.setToolTip(
            "Choose the proportion of images to retain for training.\nThe remaining images will be used for validation"
        )
        self.epoch_choice.setToolTip(
            "The number of epochs to train for.\nThe more you train, the better the model will fit the training data"
        )
        self.loss_choice.setToolTip(
            "The loss function to use for training.\nSee the list in the inference guide for more info"
        )
        self.sample_choice.setToolTip(
            "The number of samples to extract per image"
        )
        self.batch_choice.setToolTip(
            "The batch size to use for training.\n A larger value will feed more images per iteration to the model,\n"
            " which is faster and possibly improves performance, but uses more memory"
        )
        self.val_interval_choice.setToolTip(
            "The number of epochs to perform before validating data.\n "
            "The lower the value, the more often the score of the model will be computed and the more often the weights will be saved."
        )
        self.learning_rate_choice.setToolTip(
            "The learning rate to use in the optimizer. \nUse a lower value if you're using pre-trained weights"
        )
        self.augment_choice.setToolTip(
            "Check this to enable data augmentation, which will randomly deform, flip and shift the intensity in images"
            " to provide a more general dataset. \nUse this if you're extracting more than 10 samples per image"
        )
        [
            w.setToolTip("Size of the sample to extract")
            for w in self.patch_size_widgets
        ]
        self.patch_choice.setToolTip(
            "Check this to automatically crop your images in smaller, cubic images for training."
            "\nShould be used if you have a small dataset (and large images)"
        )
        self.use_deterministic_choice.setToolTip(
            "Enable deterministic training for reproducibility."
            "Using the same seed with all other parameters being similar should yield the exact same results between two runs."
        )
        self.use_transfer_choice.setToolTip(
            "Use this you want to initialize the model with pre-trained weights or use your own weights."
        )
        self.box_seed.setToolTip("Seed to use for RNG")
        ############################
        ############################

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

    def toggle_transfer_param(self):
        if self.use_transfer_choice.isChecked():
            self.custom_weights_choice.setVisible(True)
        else:
            self.custom_weights_choice.setVisible(False)

    def toggle_deterministic_param(self):
        if self.use_deterministic_choice.isChecked():
            self.container_seed.setVisible(True)
        else:
            self.container_seed.setVisible(False)

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
        data_tab, data_tab_layout = ui.make_container()
        ################
        # first group : Data
        data_group, data_layout = ui.make_group("Data")

        ui.add_widgets(
            data_layout,
            [
                ui.combine_blocks(
                    self.filetype_choice, self.lbl_filetype
                ),  # file extension
                ui.combine_blocks(
                    self.btn_image_files, self.lbl_image_files
                ),  # volumes
                ui.combine_blocks(
                    self.btn_label_files, self.lbl_label_files
                ),  # labels
                ui.combine_blocks(
                    self.btn_result_path, self.lbl_result_path
                ),  # results folder
                # ui.combine_blocks(self.model_choice, self.lbl_model_choice),  # model choice  # TODO : add custom model choice
                self.zip_choice,  # save as zip
            ],
        )

        if self.data_path != "":
            self.lbl_image_files.setText(self.data_path)

        if self.label_path != "":
            self.lbl_label_files.setText(self.label_path)

        if self.results_path != "":
            self.lbl_result_path.setText(self.results_path)

        data_group.setLayout(data_layout)
        data_tab_layout.addWidget(data_group, alignment=ui.LEFT_AL)
        # end of first group : Data
        ################
        ui.add_blank(widget=data_tab, layout=data_tab_layout)
        ################
        transfer_group_w, transfer_group_l = ui.make_group("Transfer learning")

        ui.add_widgets(
            transfer_group_l,
            [
                self.use_transfer_choice,
                self.custom_weights_choice,
                self.weights_path_container,
            ],
        )

        self.custom_weights_choice.setVisible(False)

        transfer_group_w.setLayout(transfer_group_l)
        data_tab_layout.addWidget(transfer_group_w, alignment=ui.LEFT_AL)
        ################
        ui.add_blank(self, data_tab_layout)
        ################
        ui.add_to_group(
            "Validation (%)",
            self.validation_percent_choice,
            data_tab_layout,
        )
        ################
        ui.add_blank(self, data_tab_layout)
        ################
        # buttons
        ui.add_widgets(
            data_tab_layout,
            [
                self.make_next_button(),  # next
                ui.add_blank(self),
                self.close_buttons[0],  # close
            ],
        )
        ##################
        ############
        ######
        # second tab : image sizes, data augmentation, patches size and behaviour
        ######
        ############
        ##################
        augment_tab_w, augment_tab_l = ui.make_container()
        ##################
        # extract patches or not

        patch_size_w, patch_size_l = ui.make_container()
        [
            patch_size_l.addWidget(widget, alignment=ui.LEFT_AL)
            for widgts in zip(self.patch_size_lbl, self.patch_size_widgets)
            for widget in widgts
        ]  # patch sizes
        patch_size_w.setLayout(patch_size_l)

        sampling_w, sampling_l = ui.make_container()

        ui.add_widgets(
            sampling_l,
            [
                self.lbl_sample_choice,
                self.sample_choice,  # number of samples
            ],
        )
        sampling_w.setLayout(sampling_l)
        #######################################################
        self.sampling_container = ui.combine_blocks(
            sampling_w, patch_size_w, horizontal=False, min_spacing=130, b=0
        )
        self.sampling_container.setVisible(False)
        #######################################################
        sampling = ui.combine_blocks(
            left_or_above=self.patch_choice,
            right_or_below=self.sampling_container,
            horizontal=False,
        )
        ui.add_to_group("Sampling", sampling, augment_tab_l, B=0, T=11)
        #######################
        #######################
        ui.add_blank(augment_tab_w, augment_tab_l)
        #######################
        #######################
        ui.add_to_group(
            "Augmentation",
            self.augment_choice,
            augment_tab_l,
        )
        # augment data toggle

        self.augment_choice.toggle()
        #######################
        #######################
        ui.add_blank(augment_tab_w, augment_tab_l)
        #######################
        #######################
        augment_tab_l.addWidget(
            ui.combine_blocks(
                left_or_above=self.make_prev_button(),
                right_or_below=self.make_next_button(),
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
        train_tab, train_tab_layout = ui.make_container()
        ##################
        # solo groups for loss and model
        ui.add_blank(train_tab, train_tab_layout)

        ui.add_to_group(
            "Model",
            self.model_choice,
            train_tab_layout,
        )  # model choice
        self.lbl_model_choice.setVisible(False)

        ui.add_blank(train_tab, train_tab_layout)

        ui.add_to_group(
            "Loss",
            self.loss_choice,
            train_tab_layout,
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

        ui.add_widgets(
            train_param_group_l,
            [
                ui.combine_blocks(
                    self.batch_choice,
                    self.lbl_batch_choice,
                    min_spacing=spacing,
                    horizontal=False,
                    l=5,
                    t=5,
                    r=5,
                    b=5,
                ),  # batch size
                ui.combine_blocks(
                    self.learning_rate_choice,
                    self.lbl_learning_rate_choice,
                    min_spacing=spacing,
                    horizontal=False,
                    l=5,
                    t=5,
                    r=5,
                    b=5,
                ),  # learning rate
                ui.combine_blocks(
                    self.epoch_choice,
                    self.lbl_epoch_choice,
                    min_spacing=spacing,
                    horizontal=False,
                    l=5,
                    t=5,
                    r=5,
                    b=5,
                ),  # epochs
                ui.combine_blocks(
                    self.val_interval_choice,
                    self.lbl_val_interv_choice,
                    min_spacing=spacing,
                    horizontal=False,
                    l=5,
                    t=5,
                    r=5,
                    b=5,
                ),  # validation interval
            ],
            None,
        )

        train_param_group_w.setLayout(train_param_group_l)
        train_tab_layout.addWidget(train_param_group_w)
        # end of training params group
        ##################
        ui.add_blank(self, train_tab_layout)
        ##################
        # deterministic choice group
        seed_w, seed_l = ui.make_group(
            "Deterministic training", R=1, B=5, T=11
        )
        ui.add_widgets(
            seed_l,
            [self.use_deterministic_choice, self.container_seed],
            ui.LEFT_AL,
        )

        self.container_seed.setVisible(False)

        seed_w.setLayout(seed_l)
        train_tab_layout.addWidget(seed_w)

        # end of deterministic choice group
        ##################
        # buttons

        ui.add_blank(self, train_tab_layout)

        ui.add_widgets(
            train_tab_layout,
            [
                self.make_prev_button(),  # previous
                self.btn_start,  # start
                ui.add_blank(self),
                self.close_buttons[2],
            ],
        )
        ##################
        ############
        ######
        # end of tab layouts

        ui.ScrollArea.make_scrollable(
            contained_layout=data_tab_layout,
            parent=data_tab,
            min_wh=[200, 300],
        )  # , max_wh=[200,1000])

        ui.ScrollArea.make_scrollable(
            contained_layout=augment_tab_l,
            parent=augment_tab_w,
            min_wh=[200, 300],
        )

        ui.ScrollArea.make_scrollable(
            contained_layout=train_tab_layout,
            parent=train_tab,
            min_wh=[200, 300],
        )
        self.addTab(data_tab, "Data")
        self.addTab(augment_tab_w, "Augmentation")
        self.addTab(train_tab, "Training")
        self.setMinimumSize(220, 100)

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

        * Fix memory allocation from torch


        Returns: Returns empty immediately if the file paths are not set correctly.

        """
        self.start_time = utils.get_time_filepath()
        self.save_as_zip = self.zip_choice.isChecked()

        if self.stop_requested:
            self.log.print_and_log("Worker is already stopping !")
            return

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
            try:
                self.data = self.create_train_dataset_dict()
            except ValueError as err:
                self.data = None
                raise err
            self.max_epochs = self.epoch_choice.value()

            validation_percent = self.validation_percent_choice.value() / 100

            print(f"val % : {validation_percent}")

            self.learning_rate = float(self.learning_rate_choice.currentText())

            seed_dict = {
                "use deterministic": self.use_deterministic_choice.isChecked(),
                "seed": self.box_seed.value(),
            }

            self.patch_size = []
            [
                self.patch_size.append(w.value())
                for w in self.patch_size_widgets
            ]

            model_dict = {
                "class": self.get_model(self.model_choice.currentText()),
                "name": self.model_choice.currentText(),
            }
            self.results_path_folder = (
                self.results_path
                + f"/{model_dict['name']}_{utils.get_date_time()}"
            )
            os.makedirs(
                self.results_path_folder, exist_ok=False
            )  # avoid overwrite where possible

            if self.use_transfer_choice.isChecked():
                if self.custom_weights_choice.isChecked():
                    weights_path = self.weights_path
                else:
                    weights_path = "use_pretrained"
            else:
                weights_path = None

            self.log.print_and_log(
                f"Saving results to : {self.results_path_folder}"
            )

            self.worker = TrainingWorker(
                device=self.get_device(),
                model_dict=model_dict,
                weights_path=weights_path,
                data_dicts=self.data,
                validation_percent=validation_percent,
                max_epochs=self.max_epochs,
                loss_function=self.get_loss(self.loss_choice.currentText()),
                learning_rate=self.learning_rate,
                val_interval=self.val_interval,
                batch_size=self.batch_size,
                results_path=self.results_path_folder,
                sampling=self.patch_choice.isChecked(),
                num_samples=self.num_samples,
                sample_size=self.patch_size,
                do_augmentation=self.augment_choice.isChecked(),
                deterministic=seed_dict,
            )
            self.worker.set_download_log(self.log)

            [btn.setVisible(False) for btn in self.close_buttons]

            self.worker.log_signal.connect(self.log.print_and_log)
            self.worker.warn_signal.connect(self.log.warn)

            self.worker.started.connect(self.on_start)

            self.worker.yielded.connect(
                lambda data: self.on_yield(data, widget=self)
            )
            self.worker.finished.connect(self.on_finish)

            self.worker.errored.connect(self.on_error)

        if self.worker.is_running:
            self.log.print_and_log("*" * 20)
            self.log.print_and_log(
                f"Stop requested at {utils.get_time()}. \nWaiting for next yielding step..."
            )
            self.stop_requested = True
            self.btn_start.setText("Stopping... Please wait")
            self.log.print_and_log("*" * 20)
            self.worker.quit()
        else:
            self.worker.start()
            self.btn_start.setText("Running...  Click to stop")

    def on_start(self):
        """Catches started signal from worker"""

        self.remove_docked_widgets()
        self.display_status_report()

        self.log.print_and_log(f"Worker started at {utils.get_time()}")
        self.log.print_and_log("\nWorker is running...")

    def on_finish(self):
        """Catches finished signal from worker"""
        self.log.print_and_log("*" * 20)
        self.log.print_and_log(f"\nWorker finished at {utils.get_time()}")

        self.log.print_and_log(f"Saving in {self.results_path_folder}")
        self.log.print_and_log(f"Saving last loss plot")

        if self.canvas is not None:
            self.canvas.figure.savefig(
                (
                    self.results_path_folder
                    + f"/final_metric_plots_{utils.get_time_filepath()}.png"
                ),
                format="png",
            )

        self.log.print_and_log("Saving log")
        self.save_log_to_path(self.results_path_folder)

        self.log.print_and_log("Done")
        self.log.print_and_log("*" * 10)

        self.make_csv()

        self.btn_start.setText("Start")
        [btn.setVisible(True) for btn in self.close_buttons]

        del self.worker
        self.worker = None
        self.empty_cuda_cache()

        if self.save_as_zip:
            shutil.make_archive(
                self.results_path_folder, "zip", self.results_path_folder
            )

        # if zipfile.is_zipfile(self.results_path_folder+".zip"):

        # if not shutil.rmtree.avoids_symlink_attacks:
        #     raise RuntimeError("shutil.rmtree is not safe on this platform")

        # shutil.rmtree(self.results_path_folder)

        self.results_path_folder = ""

        # self.clean_cache() # trying to fix memory leak

    def on_error(self):
        """Catches errored signal from worker"""
        self.log.print_and_log(f"WORKER ERRORED at {utils.get_time()}")
        self.worker = None
        self.empty_cuda_cache()
        # self.clean_cache()

    @staticmethod
    def on_yield(data, widget):
        # print(
        #     f"\nCatching results : for epoch {data['epoch']},
        #     loss is {data['losses']} and validation is {data['val_metrics']}"
        # )
        if data["plot"]:
            widget.progress.setValue(
                100 * (data["epoch"] + 1) // widget.max_epochs
            )

            widget.update_loss_plot(data["losses"], data["val_metrics"])
            widget.loss_values = data["losses"]
            widget.validation_values = data["val_metrics"]

        if widget.stop_requested:
            widget.log.print_and_log(
                "Saving weights from aborted training in results folder"
            )
            torch.save(
                data["weights"],
                os.path.join(
                    widget.results_path_folder,
                    f"latest_weights_aborted_training_{utils.get_time_filepath()}.pth",
                ),
            )
            widget.log.print_and_log("Saving complete")
            widget.stop_requested = False

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

    def make_csv(self):

        size_column = range(1, self.max_epochs + 1)

        if len(self.loss_values) == 0 or self.loss_values is None:
            warnings.warn("No loss values to add to csv !")
            return

        self.df = pd.DataFrame(
            {
                "epoch": size_column,
                "loss": self.loss_values,
                "validation": utils.fill_list_in_between(
                    self.validation_values, self.val_interval - 1, ""
                )[: len(size_column)],
            }
        )
        path = os.path.join(self.results_path_folder, "training.csv")
        self.df.to_csv(path, index=False)

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

            plot_path = self.results_path_folder + "/Loss_plots"
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
