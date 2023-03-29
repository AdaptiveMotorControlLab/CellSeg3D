import shutil
import warnings
from functools import partial
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
from napari_cellseg3d import config
from napari_cellseg3d import interface as ui
from napari_cellseg3d import utils
from napari_cellseg3d.code_models.model_framework import ModelFramework
from napari_cellseg3d.code_models.model_workers import TrainingReport
from napari_cellseg3d.code_models.model_workers import TrainingWorker

NUMBER_TABS = 3
DEFAULT_PATCH_SIZE = 64

logger = utils.LOGGER


class Trainer(ModelFramework, metaclass=ui.QWidgetSingleton):
    """A plugin to train pre-defined PyTorch models for one-channel segmentation directly in napari.
    Features parameter selection for training, dynamic loss plotting and automatic saving of the best weights during
    training through validation."""

    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
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
        self.enable_utils_menu()

        self.data_path = None
        self.label_path = None
        self.results_path = None
        """Path to the folder inside the results path that contains all results"""

        self.config = config.TrainerConfig()

        self.model = None  # TODO : custom model loading ?
        self.worker = None
        """Training worker for multithreading, should be a TrainingWorker instance from :doc:model_workers.py"""
        self.worker_config = None
        self.data = None
        """Data dictionary containing file paths"""
        self.stop_requested = False
        """Whether the worker should stop or not"""
        self.start_time = None

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
        self.result_layers = []
        """Layers to display checkpoint"""

        self.df = None
        self.loss_values = []
        self.validation_values = []

        # self.model_choice.setCurrentIndex(0)

        ################################
        # interface
        default = config.TrainingWorkerConfig()

        self.zip_choice = ui.CheckBox("Compress results")

        self.validation_percent_choice = ui.Slider(
            lower=10,
            upper=90,
            default=default.validation_percent * 100,
            step=5,
            parent=self,
        )

        self.epoch_choice = ui.IntIncrementCounter(
            lower=2,
            upper=200,
            default=default.max_epochs,
            label="Number of epochs : ",
        )

        self.loss_choice = ui.DropdownMenu(
            sorted(self.loss_dict.keys()), label="Loss function"
        )
        self.lbl_loss_choice = self.loss_choice.label
        self.loss_choice.setCurrentIndex(0)

        self.sample_choice_slider = ui.Slider(
            lower=2,
            upper=50,
            default=default.num_samples,
            text_label="Number of patches per image : ",
        )

        self.sample_choice_slider.container.setVisible(False)

        self.batch_choice = ui.Slider(
            lower=1,
            upper=10,
            default=default.batch_size,
            text_label="Batch size : ",
        )

        self.val_interval_choice = ui.IntIncrementCounter(
            default=default.validation_interval,
            label="Validation interval : ",
        )

        self.epoch_choice.valueChanged.connect(self._update_validation_choice)
        self.val_interval_choice.valueChanged.connect(
            self._update_validation_choice
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

        self.augment_choice = ui.CheckBox("Augment data")

        self.close_buttons = [
            self._make_close_button() for i in range(NUMBER_TABS)
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
        self.sampling_container = ui.ContainerWidget()

        self.patch_choice = ui.CheckBox(
            "Extract patches from images", func=self._toggle_patch_dims
        )
        self.patch_choice.clicked.connect(self._toggle_patch_dims)

        self.use_transfer_choice = ui.CheckBox(
            "Transfer weights", self._toggle_transfer_param
        )

        self.use_deterministic_choice = ui.CheckBox(
            "Deterministic training", func=self._toggle_deterministic_param
        )
        self.box_seed = ui.IntIncrementCounter(
            upper=10000000, default=default.deterministic_config.seed
        )
        self.lbl_seed = ui.make_label("Seed", self)
        self.container_seed = ui.combine_blocks(
            self.box_seed, self.lbl_seed, horizontal=False
        )

        self.progress.setVisible(False)
        """Dock widget containing the progress bar"""

        self.btn_start = ui.Button("Start training", self.start)

        # self.btn_model_path.setVisible(False)
        # self.lbl_model_path.setVisible(False)

        ############################
        ############################
        def set_tooltips():
            # tooltips
            self.zip_choice.setToolTip(
                "Checking this will save a copy of the results as a zip folder"
            )
            self.validation_percent_choice.tooltips = "Choose the proportion of images to retain for training.\nThe remaining images will be used for validation"
            self.epoch_choice.tooltips = "The number of epochs to train for.\nThe more you train, the better the model will fit the training data"
            self.loss_choice.setToolTip(
                "The loss function to use for training.\nSee the list in the inference guide for more info"
            )
            self.sample_choice_slider.tooltips = (
                "The number of samples to extract per image"
            )
            self.batch_choice.tooltips = (
                "The batch size to use for training.\n A larger value will feed more images per iteration to the model,\n"
                " which is faster and possibly improves performance, but uses more memory"
            )
            self.val_interval_choice.tooltips = (
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
        set_tooltips()
        self._build()

    def _hide_unused(self):
        [
            self._hide_io_element(w)
            for w in [
                self.layer_choice,
                self.folder_choice,
                self.label_layer_loader,
                self.image_layer_loader,
            ]
        ]

    def _update_validation_choice(self):
        validation = self.val_interval_choice
        max_epoch = self.epoch_choice.value()

        if validation.value() > max_epoch:
            validation.setValue(max_epoch)
            validation.setMaximum(max_epoch)
        elif validation.maximum() < max_epoch:
            validation.setMaximum(max_epoch)

    def get_loss(self, key):
        """Getter for loss function selected by user"""
        return self.loss_dict[key]

    def _toggle_patch_dims(self):
        if self.patch_choice.isChecked():
            [w.setVisible(True) for w in self.patch_size_widgets]
            [l.setVisible(True) for l in self.patch_size_lbl]
            self.sample_choice_slider.container.setVisible(True)
            self.sampling_container.setVisible(True)
        else:
            [w.setVisible(False) for w in self.patch_size_widgets]
            [l.setVisible(False) for l in self.patch_size_lbl]
            self.sample_choice_slider.container.setVisible(False)
            self.sampling_container.setVisible(False)

    def _toggle_transfer_param(self):
        if self.use_transfer_choice.isChecked():
            self.custom_weights_choice.setVisible(True)
        else:
            self.custom_weights_choice.setVisible(False)

    def _toggle_deterministic_param(self):
        if self.use_deterministic_choice.isChecked():
            self.container_seed.setVisible(True)
        else:
            self.container_seed.setVisible(False)

    def check_ready(self):
        """
        Checks that the paths to the images and labels are correctly set

        Returns:

            * True if paths are set correctly

            * False and displays a warning if not

        """
        if self.images_filepaths != [] and self.labels_filepaths != []:
            return True
        else:
            warnings.formatwarning = utils.format_Warning
            warnings.warn("Image and label paths are not correctly set")
            return False

    def _build(self):
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

        # for w in self.children():
        #     w.setToolTip(f"{w}")

        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.MinimumExpanding)
        ########
        ################
        ########################
        # first tab : model and dataset choices
        data_tab = ui.ContainerWidget()
        ################
        # first group : Data
        data_group, data_layout = ui.make_group("Data")

        ui.add_widgets(
            data_layout,
            [
                ui.combine_blocks(
                    self.filetype_choice, self.filetype_choice.label
                ),  # file extension
                self.image_filewidget,
                self.labels_filewidget,
                self.results_filewidget,
                # ui.combine_blocks(self.model_choice, self.model_choice.label), # model choice
                # TODO : add custom model choice
                self.zip_choice,  # save as zip
            ],
        )

        for w in [
            self.image_filewidget,
            self.labels_filewidget,
            self.results_filewidget,
        ]:
            w.check_ready()

        if self.data_path is not None:
            self.image_filewidget.text_field.setText(self.data_path)

        if self.label_path is not None:
            self.labels_filewidget.text_field.setText(self.label_path)

        if self.results_path is not None:
            self.results_filewidget.text_field.setText(self.results_path)

        data_group.setLayout(data_layout)
        data_tab.layout.addWidget(data_group, alignment=ui.LEFT_AL)
        # end of first group : Data
        ################
        ui.add_blank(widget=data_tab, layout=data_tab.layout)
        ################
        transfer_group_w, transfer_group_l = ui.make_group("Transfer learning")

        ui.add_widgets(
            transfer_group_l,
            [
                self.use_transfer_choice,
                self.custom_weights_choice,
                self.weights_filewidget,
            ],
        )
        self.custom_weights_choice.setVisible(False)
        self.weights_filewidget.setVisible(False)

        transfer_group_w.setLayout(transfer_group_l)
        data_tab.layout.addWidget(transfer_group_w, alignment=ui.LEFT_AL)
        ################
        ui.add_blank(self, data_tab.layout)
        ################
        ui.GroupedWidget.create_single_widget_group(
            "Validation (%)",
            self.validation_percent_choice.container,
            data_tab.layout,
        )
        ################
        ui.add_blank(self, data_tab.layout)
        ################
        # buttons
        ui.add_widgets(
            data_tab.layout,
            [
                self._make_next_button(),  # next
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
        augment_tab_w = ui.ContainerWidget()
        augment_tab_l = augment_tab_w.layout
        ##################
        # extract patches or not

        patch_size_w = ui.ContainerWidget()
        patch_size_l = patch_size_w.layout
        [
            patch_size_l.addWidget(widget, alignment=ui.LEFT_AL)
            for widgts in zip(self.patch_size_lbl, self.patch_size_widgets)
            for widget in widgts
        ]  # patch sizes
        patch_size_w.setLayout(patch_size_l)

        sampling_w = ui.ContainerWidget()
        sampling_l = sampling_w.layout

        ui.add_widgets(
            sampling_l,
            [
                self.sample_choice_slider.container,  # number of samples
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
        ui.GroupedWidget.create_single_widget_group(
            "Sampling", sampling, augment_tab_l, b=0, t=11
        )
        #######################
        #######################
        ui.add_blank(augment_tab_w, augment_tab_l)
        #######################
        #######################
        ui.GroupedWidget.create_single_widget_group(
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
                left_or_above=self._make_prev_button(),
                right_or_below=self._make_next_button(),
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
        train_tab = ui.ContainerWidget()
        ##################
        # solo groups for loss and model
        ui.add_blank(train_tab, train_tab.layout)

        ui.GroupedWidget.create_single_widget_group(
            "Model",
            self.model_choice,
            train_tab.layout,
        )  # model choice
        self.model_choice.label.setVisible(False)

        ui.add_blank(train_tab, train_tab.layout)
        ui.GroupedWidget.create_single_widget_group(
            "Loss",
            self.loss_choice,
            train_tab.layout,
        )  # loss choice
        self.lbl_loss_choice.setVisible(False)

        # end of solo groups for loss and model
        ##################
        ui.add_blank(train_tab, train_tab.layout)
        ##################
        # training params group

        train_param_group_w, train_param_group_l = ui.make_group(
            "Training parameters", r=1, b=5, t=11
        )

        spacing = 20

        ui.add_widgets(
            train_param_group_l,
            [
                self.batch_choice.container,  # batch size
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
                self.epoch_choice.label,  # epochs
                self.epoch_choice,
                self.val_interval_choice.label,
                self.val_interval_choice,  # validation interval
            ],
            None,
        )

        train_param_group_w.setLayout(train_param_group_l)
        train_tab.layout.addWidget(train_param_group_w)
        # end of training params group
        ##################
        ui.add_blank(self, train_tab.layout)
        ##################
        # deterministic choice group
        seed_w, seed_l = ui.make_group(
            "Deterministic training", r=1, b=5, t=11
        )
        ui.add_widgets(
            seed_l,
            [self.use_deterministic_choice, self.container_seed],
            ui.LEFT_AL,
        )

        self.container_seed.setVisible(False)

        seed_w.setLayout(seed_l)
        train_tab.layout.addWidget(seed_w)

        # end of deterministic choice group
        ##################
        # buttons

        ui.add_blank(self, train_tab.layout)

        ui.add_widgets(
            train_tab.layout,
            [
                self._make_prev_button(),  # previous
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
            contained_layout=data_tab.layout,
            parent=data_tab,
            min_wh=[200, 300],
        )  # , max_wh=[200,1000])

        ui.ScrollArea.make_scrollable(
            contained_layout=augment_tab_l,
            parent=augment_tab_w,
            min_wh=[200, 300],
        )

        ui.ScrollArea.make_scrollable(
            contained_layout=train_tab.layout,
            parent=train_tab,
            min_wh=[200, 300],
        )
        self.addTab(data_tab, "Data")
        self.addTab(augment_tab_w, "Augmentation")
        self.addTab(train_tab, "Training")
        self.setMinimumSize(220, 100)

        self._hide_unused()

        default_results_path = (
            config.TrainingWorkerConfig().results_path_folder
        )
        self.results_filewidget.text_field.setText(default_results_path)
        self.results_filewidget.check_ready()
        self._check_results_path(default_results_path)
        self.results_path = default_results_path

    # def _show_dialog_lab(self):
    #     """Shows the  dialog to load label files in a path, loads them (see :doc:model_framework) and changes the widget
    #     label :py:attr:`self.label_filewidget.text_field` accordingly"""
    #     folder = ui.open_folder_dialog(self, self._default_path)
    #
    #     if folder:
    #         self.label_path = folder
    #         self.labels_filewidget.text_field.setText(self.label_path)

    def send_log(self, text):
        """Sends a message via the Log attribute"""
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

            self._reset_loss_plot()

            try:
                self.data = self.create_train_dataset_dict()
            except ValueError as err:
                self.data = None
                raise err

            model_config = config.ModelInfo(
                name=self.model_choice.currentText()
            )

            self.weights_config.path = self.weights_config.path
            self.weights_config.custom = self.custom_weights_choice.isChecked()
            self.weights_config.use_pretrained = (
                not self.use_transfer_choice.isChecked()
            )

            deterministic_config = config.DeterministicConfig(
                enabled=self.use_deterministic_choice.isChecked(),
                seed=self.box_seed.value(),
            )

            validation_percent = (
                self.validation_percent_choice.slider_value / 100
            )

            results_path_folder = Path(
                self.results_path
                + f"/{model_config.name}_{utils.get_date_time()}"
            )
            Path(results_path_folder).mkdir(
                parents=True, exist_ok=False
            )  # avoid overwrite where possible

            patch_size = [w.value() for w in self.patch_size_widgets]

            logger.debug("Loading config...")
            self.worker_config = config.TrainingWorkerConfig(
                device=self.get_device(),
                model_info=model_config,
                weights_info=self.weights_config,
                train_data_dict=self.data,
                validation_percent=validation_percent,
                max_epochs=self.epoch_choice.value(),
                loss_function=self.get_loss(self.loss_choice.currentText()),
                learning_rate=float(self.learning_rate_choice.currentText()),
                validation_interval=self.val_interval_choice.value(),
                batch_size=self.batch_choice.slider_value,
                results_path_folder=str(results_path_folder),
                sampling=self.patch_choice.isChecked(),
                num_samples=self.sample_choice_slider.slider_value,
                sample_size=patch_size,
                do_augmentation=self.augment_choice.isChecked(),
                deterministic_config=deterministic_config,
            )  # TODO(cyril) continue to put params in config

            self.config = config.TrainerConfig(
                save_as_zip=self.zip_choice.isChecked()
            )

            self.log.print_and_log(
                f"Saving results to : {results_path_folder}"
            )

            self.worker = TrainingWorker(config=self.worker_config)
            self.worker.set_download_log(self.log)

            [btn.setVisible(False) for btn in self.close_buttons]

            self.worker.log_signal.connect(self.log.print_and_log)
            self.worker.warn_signal.connect(self.log.warn)

            self.worker.started.connect(self.on_start)

            self.worker.yielded.connect(partial(self.on_yield))
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

        self.log.print_and_log(
            f"Saving in {self.worker_config.results_path_folder}"
        )
        self.log.print_and_log(f"Saving last loss plot")

        plot_name = self.worker_config.results_path_folder / Path(
            f"final_metric_plots_{utils.get_time_filepath()}.png"
        )
        if self.canvas is not None:
            self.canvas.figure.savefig(
                plot_name,
                format="png",
            )

        self.log.print_and_log("Saving log")
        self.save_log_to_path(self.worker_config.results_path_folder)

        self.log.print_and_log("Done")
        self.log.print_and_log("*" * 10)

        self._make_csv()

        self.btn_start.setText("Start")
        [btn.setVisible(True) for btn in self.close_buttons]

        # del self.worker

        # self.empty_cuda_cache()

        if self.config.save_as_zip:
            shutil.make_archive(
                self.worker_config.results_path_folder,
                "zip",
                self.worker_config.results_path_folder,
            )

        self.worker = None
        # if zipfile.is_zipfile(self.results_path_folder+".zip"):

        # if not shutil.rmtree.avoids_symlink_attacks:
        #     raise RuntimeError("shutil.rmtree is not safe on this platform")

        # shutil.rmtree(self.results_path_folder)

        # self.results_path_folder = ""

        # self.clean_cache() # trying to fix memory leak

    def on_error(self):
        """Catches errored signal from worker"""
        self.log.print_and_log(f"WORKER ERRORED at {utils.get_time()}")
        self.worker = None
        # self.empty_cuda_cache()
        # self.clean_cache()

    def on_yield(self, report: TrainingReport):
        # logger.info(
        #     f"\nCatching results : for epoch {data['epoch']},
        #     loss is {data['losses']} and validation is {data['val_metrics']}"
        # )
        if report == TrainingReport():
            return

        if report.show_plot:

            try:
                layer_name = "Training_checkpoint_"
                rge = range(len(report.images))

                self.log.print_and_log(len(report.images))

                if report.epoch + 1 == self.worker_config.validation_interval:
                    for i in rge:
                        layer = self._viewer.add_image(
                            report.images[i],
                            name=layer_name + str(i),
                            colormap="twilight",
                        )
                        self.result_layers.append(layer)
                else:
                    for i in rge:
                        if layer_name + str(i) not in [
                            layer.name for layer in self.result_layers
                        ]:
                            new_layer = self._viewer.add_image(
                                report.images[i],
                                name=layer_name + str(i),
                                colormap="twilight",
                            )
                            self.result_layers.append(new_layer)
                        self.result_layers[i].data = report.images[i]
                        self.result_layers[i].refresh()
            except Exception as e:
                logger.error(e)

            self.progress.setValue(
                100 * (report.epoch + 1) // self.worker_config.max_epochs
            )

            self.update_loss_plot(report.loss_values, report.validation_metric)
            self.loss_values = report.loss_values
            self.validation_values = report.validation_metric

        if self.stop_requested:
            self.log.print_and_log(
                "Saving weights from aborted training in results folder"
            )
            torch.save(
                report.weights,
                Path(self.worker_config.results_path_folder)
                / Path(
                    f"latest_weights_aborted_training_{utils.get_time_filepath()}.pth",
                ),
            )
            self.log.print_and_log("Saving complete")
            self.stop_requested = False

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

    def _make_csv(self):

        size_column = range(1, self.worker_config.max_epochs + 1)

        if len(self.loss_values) == 0 or self.loss_values is None:
            warnings.warn("No loss values to add to csv !")
            return

        self.df = pd.DataFrame(
            {
                "epoch": size_column,
                "loss": self.loss_values,
                "validation": utils.fill_list_in_between(
                    self.validation_values,
                    self.worker_config.validation_interval - 1,
                    "",
                )[: len(size_column)],
            }
        )
        path = Path(self.worker_config.results_path_folder) / Path(
            "training.csv"
        )
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
            x = [
                self.worker_config.validation_interval * (i + 1)
                for i in range(len(dice_metric))
            ]
            y = dice_metric

            epoch_min = (
                np.argmax(y) + 1
            ) * self.worker_config.validation_interval
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

            plot_path = self.worker_config.results_path_folder / Path(
                "../Loss_plots"
            )
            Path(plot_path).mkdir(parents=True, exist_ok=True)

            if self.canvas is not None:
                self.canvas.figure.savefig(
                    str(
                        plot_path
                        / f"checkpoint_metric_plots_{utils.get_date_time()}.png"
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
        if epoch < self.worker_config.validation_interval * 2:
            return
        elif epoch == self.worker_config.validation_interval * 2:
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
            try:
                self.plot_dock = self._viewer.window.add_dock_widget(
                    self.canvas, name="Loss plots", area="bottom"
                )
                self.plot_dock._close_btn = False
            except AttributeError as e:
                logger.error(e)
                logger.error(
                    "Plot dock widget could not be added. Should occur in testing only"
                )

            self.docked_widgets.append(self.plot_dock)
            self.plot_loss(loss, metric)
        else:
            with plt.style.context("dark_background"):

                self.train_loss_plot.cla()
                self.dice_metric_plot.cla()

                self.plot_loss(loss, metric)

    def _reset_loss_plot(self):
        if (
            self.train_loss_plot is not None
            and self.dice_metric_plot is not None
        ):
            with plt.style.context("dark_background"):
                self.train_loss_plot.cla()
                self.dice_metric_plot.cla()
