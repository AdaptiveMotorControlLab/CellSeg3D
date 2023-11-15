"""Training plugin for napari_cellseg3d."""

import shutil
import warnings
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.figure import Figure

if TYPE_CHECKING:
    import napari

# Qt
from qtpy.QtWidgets import QSizePolicy

# local
from napari_cellseg3d import config, utils
from napari_cellseg3d import interface as ui
from napari_cellseg3d.code_models.model_framework import ModelFramework
from napari_cellseg3d.code_models.worker_training import (
    SupervisedTrainingWorker,
    WNetTrainingWorker,
)
from napari_cellseg3d.code_models.workers_utils import TrainingReport

logger = utils.LOGGER
NUMBER_TABS = 4  # how many tabs in the widget
DEFAULT_PATCH_SIZE = 64  # default patch size for training


class Trainer(ModelFramework, metaclass=ui.QWidgetSingleton):
    """A plugin to train pre-defined PyTorch models for one-channel segmentation directly in napari.

    Features parameter selection for training, dynamic loss plotting and automatic saving of the best weights during
    training through validation.
    """

    default_config = config.SupervisedTrainingWorkerConfig()

    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
    ):
        """Creates a Trainer tab widget with the following functionalities.

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

        self.model = None
        self.worker = None
        """Training worker for multithreading, should be a TrainingWorker instance from :doc:model_workers.py"""
        self.worker_config = None
        self.data = None
        """Data dictionary containing file paths"""
        self._stop_requested = False
        """Whether the worker should stop or not"""
        self.start_time = None
        """Start time of the latest job"""
        self.unsupervised_mode = False
        self.unsupervised_eval_data = None

        self.loss_list = [  # MUST BE MATCHED WITH THE LOSS FUNCTIONS IN THE TRAINING WORKER DICT
            "Dice",
            "Generalized Dice",
            "DiceCE",
            "Tversky",
            # "Focal loss",
            # "Dice-Focal loss",
        ]
        """List of loss functions"""

        self.canvas = None
        """Canvas to plot loss and dice metric in"""
        self.plot_1 = None
        """Plot for loss"""
        self.plot_2 = None
        """Plot for dice metric"""
        self.plot_dock = None
        """Docked widget with plots"""
        self.result_layers: List[napari.layers.Layer] = []
        """Layers to display checkpoint"""

        self.plot_1_labels = {
            "title": {
                "supervised": "Epoch average loss",
                "unsupervised": "Metrics",
            },
            "ylabel": {
                "supervised": "Loss",
                "unsupervised": "",
            },
        }
        self.plot_2_labels = {
            "title": {
                "supervised": "Epoch average dice metric",
                "unsupervised": "Reconstruction loss",
            },
            "ylabel": {
                "supervised": "Metric",
                "unsupervised": "Loss",
            },
        }

        self.df = None
        self.loss_1_values = {}
        self.loss_2_values = []

        ###########
        # interface
        ###########

        self.zip_choice = ui.CheckBox("Compress results")
        self.train_split_percent_choice = ui.Slider(
            lower=10,
            upper=90,
            default=self.default_config.training_percent * 100,
            step=5,
            parent=self,
        )

        self.epoch_choice = ui.IntIncrementCounter(
            lower=2,
            upper=999,
            default=self.default_config.max_epochs,
            text_label="Number of epochs : ",
        )

        self.loss_choice = ui.DropdownMenu(
            self.loss_list,
            text_label="Loss function",
        )
        self.lbl_loss_choice = self.loss_choice.label
        self.loss_choice.setCurrentIndex(0)

        self.sample_choice_slider = ui.Slider(
            lower=2,
            upper=50,
            default=self.default_config.num_samples,
            text_label="Number of patches per image : ",
        )

        self.sample_choice_slider.container.setVisible(False)

        self.batch_choice = ui.Slider(
            lower=1,
            upper=10,
            default=self.default_config.batch_size,
            text_label="Batch size : ",
        )

        self.val_interval_choice = ui.IntIncrementCounter(
            default=self.default_config.validation_interval,
            text_label="Validation interval : ",
        )

        self.epoch_choice.valueChanged.connect(self._update_validation_choice)
        self.val_interval_choice.valueChanged.connect(
            self._update_validation_choice
        )

        self.learning_rate_choice = LearningRateWidget(parent=self)
        self.lbl_learning_rate_choice = (
            self.learning_rate_choice.lr_value_choice.label
        )

        self.scheduler_patience_choice = ui.IntIncrementCounter(
            1,
            99,
            default=self.default_config.scheduler_patience,
            text_label="Scheduler patience",
        )
        self.scheduler_factor_choice = ui.Slider(
            divide_factor=100,
            default=self.default_config.scheduler_factor * 100,
            text_label="Scheduler factor :",
        )

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
            upper=1000000000,
            default=self.default_config.deterministic_config.seed,
        )
        self.lbl_seed = ui.make_label("Seed", self)
        self.container_seed = ui.combine_blocks(
            self.box_seed, self.lbl_seed, horizontal=False
        )

        self.progress.setVisible(False)
        """Dock widget containing the progress bar"""

        # widgets created later and only shown if supervised model is selected
        self.start_button_supervised = None
        self.loss_group = None
        self.validation_group = None
        ############################
        ############################
        # WNet parameters
        self.wnet_widgets = (
            None  # widgets created later and only shown if WNet is selected
        )
        self.advanced_next_button = (
            None  # button created later and only shown if WNet is selected
        )
        self.start_button_unsupervised = (
            None  # button created later and only shown if WNet is selected
        )
        ############################
        # self.btn_model_path.setVisible(False)
        # self.lbl_model_path.setVisible(False)
        ############################
        ############################
        self._set_tooltips()
        self._build()

        self.model_choice.setCurrentIndex(4)
        self.model_choice.currentTextChanged.connect(
            partial(self._toggle_unsupervised_mode, enabled=False)
        )
        self._toggle_unsupervised_mode(enabled=True)

    def _set_tooltips(self):
        # tooltips
        self.zip_choice.setToolTip(
            "Save a copy of the results as a zip folder"
        )
        self.train_split_percent_choice.tooltips = "The percentage of images to retain for training.\nThe remaining images will be used for validation"
        self.epoch_choice.tooltips = "The number of epochs to train for.\nThe more you train, the better the model will fit the training data"
        self.loss_choice.setToolTip(
            "The loss function to use for training.\nSee the list in the training guide for more info"
        )
        self.sample_choice_slider.tooltips = (
            "The number of samples to extract per image"
        )
        self.batch_choice.tooltips = (
            "The batch size to use for training.\n A larger value will feed more images per iteration to the model,\n"
            " which is faster and can improve performance, but uses more memory on your selected device"
        )
        self.val_interval_choice.tooltips = (
            "The number of epochs to perform before validating on test data.\n "
            "The lower the value, the more often the score of the model will be computed and the more often the weights will be saved."
        )
        self.learning_rate_choice.setToolTip(
            "The learning rate to use in the optimizer. \nUse a lower value if you're using pre-trained weights"
        )
        self.scheduler_factor_choice.setToolTip(
            "The factor by which to reduce the learning rate once the loss reaches a plateau"
        )
        self.scheduler_patience_choice.setToolTip(
            "The amount of epochs to wait for before reducing the learning rate"
        )
        self.augment_choice.setToolTip(
            "Check this to enable data augmentation, which will randomly deform, flip and shift the intensity in images"
            " to provide a more diverse dataset"
        )
        [
            w.setToolTip("Size of the sample to extract")
            for w in self.patch_size_widgets
        ]
        self.patch_choice.setToolTip(
            "Check this to automatically crop your images into smaller, cubic images for training."
            "\nShould be used if you have a few large images"
        )
        self.use_deterministic_choice.setToolTip(
            "Enable deterministic training for reproducibility."
            "Using the same seed with all other parameters being similar should yield the exact same results across runs."
        )
        self.use_transfer_choice.setToolTip(
            "Use this you want to initialize the model with pre-trained weights or use your own weights."
        )
        self.box_seed.setToolTip("Seed to use for RNG")

    def _make_start_button(self):
        return ui.Button("Start training", self.start, parent=self)

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
            self.weights_filewidget.setVisible(False)

    def _toggle_deterministic_param(self):
        if self.use_deterministic_choice.isChecked():
            self.container_seed.setVisible(True)
        else:
            self.container_seed.setVisible(False)

    def check_ready(self):
        """Checks that the paths to the images and labels are correctly set.

        Returns:
            * True if paths are set correctly

            * False and displays a warning if not

        """
        if not self.unsupervised_mode:
            if (
                self.images_filepaths == []
                or self.labels_filepaths == []
                or len(self.images_filepaths) != len(self.labels_filepaths)
            ):
                logger.warning("Image and label paths are not correctly set")
                return False
        else:
            if self.get_unsupervised_image_filepaths() == []:
                logger.warning("Image paths are not correctly set")
                return False
        return True

    def _toggle_unsupervised_mode(self, enabled=False):
        """Change all the UI elements needed for unsupervised learning mode."""
        if self.model_choice.currentText() == "WNet" or enabled:
            unsupervised = True
            self.start_btn = self.start_button_unsupervised
            if self.image_filewidget.text_field.text() == "Images directory":
                self.image_filewidget.text_field.setText("Validation images")
            if self.labels_filewidget.text_field.text() == "Labels directory":
                self.labels_filewidget.text_field.setText("Validation labels")
            self.learning_rate_choice.lr_value_choice.setValue(2)
            self.learning_rate_choice.lr_exponent_choice.setCurrentIndex(3)
        else:
            unsupervised = False
            self.start_btn = self.start_button_supervised
            if self.image_filewidget.text_field.text() == "Validation images":
                self.image_filewidget.text_field.setText("Images directory")
            if self.labels_filewidget.text_field.text() == "Validation labels":
                self.labels_filewidget.text_field.setText("Labels directory")
            self.learning_rate_choice.lr_value_choice.setValue(1)
            self.learning_rate_choice.lr_exponent_choice.setCurrentIndex(1)

        supervised = not unsupervised
        self.unsupervised_mode = unsupervised

        self.setTabVisible(3, unsupervised)
        self.setTabEnabled(3, unsupervised)
        self.start_button_unsupervised.setVisible(unsupervised)
        self.start_button_supervised.setVisible(supervised)
        self.advanced_next_button.setVisible(unsupervised)
        # loss
        # self.loss_choice.setVisible(supervised)
        self.loss_group.setVisible(supervised)
        # scheduler
        self.scheduler_factor_choice.container.setVisible(supervised)
        self.scheduler_factor_choice.label.setVisible(supervised)
        self.scheduler_patience_choice.setVisible(supervised)
        self.scheduler_patience_choice.label.setVisible(supervised)
        # data
        self.unsupervised_images_filewidget.setVisible(unsupervised)
        self.validation_group.setVisible(supervised)
        self.image_filewidget.required = supervised
        self.labels_filewidget.required = supervised

        self._check_all_filepaths()

    def _build(self):
        """Builds the layout of the widget and creates the following tabs and prompts.

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

        * Start (see :py:func:`~start`)
        """
        # for w in self.children():
        #     w.setToolTip(f"{w}")

        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.MinimumExpanding)
        ########
        ################
        ########################
        # first tab : model, weights and device choices
        model_tab = ui.ContainerWidget()
        ################
        ui.GroupedWidget.create_single_widget_group(
            "Model",
            self.model_choice,
            model_tab.layout,
        )  # model choice
        self.model_choice.label.setVisible(False)
        ui.add_blank(model_tab, model_tab.layout)
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
        model_tab.layout.addWidget(transfer_group_w, alignment=ui.LEFT_AL)
        ################
        ui.add_blank(self, model_tab.layout)
        ################
        ui.GroupedWidget.create_single_widget_group(
            "Device",
            self.device_choice,
            model_tab.layout,
        )
        ################
        ui.add_blank(self, model_tab.layout)
        ################
        # buttons
        ui.add_widgets(
            model_tab.layout,
            [
                self._make_next_button(),  # next
                ui.add_blank(self),
                self.close_buttons[0],  # close
            ],
        )
        ##################
        ############
        ######
        # Second tab : image sizes, data augmentation, patches size and behaviour
        ######
        ############
        ##################
        data_tab_w = ui.ContainerWidget()
        data_tab_l = data_tab_w.layout
        ##################
        ################
        # group : Data
        data_group, data_layout = ui.make_group("Data")

        ui.add_widgets(
            data_layout,
            [
                self.unsupervised_images_filewidget,
                self.image_filewidget,
                self.labels_filewidget,
                ui.make_label("Results :", parent=self),
                self.results_filewidget,
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
        data_tab_l.addWidget(data_group, alignment=ui.LEFT_AL)
        # end of first group : Data
        ################
        ui.add_blank(widget=data_tab_w, layout=data_tab_l)
        ################
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
            "Sampling", sampling, data_tab_l, b=0, t=11
        )
        #######################
        #######################
        ui.add_blank(data_tab_w, data_tab_l)
        #######################
        #######################
        ui.GroupedWidget.create_single_widget_group(
            "Augmentation",
            self.augment_choice,
            data_tab_l,
        )
        # augment data toggle

        self.augment_choice.toggle()
        #######################
        ui.add_blank(data_tab_w, data_tab_l)
        #######################
        self.validation_group = ui.GroupedWidget.create_single_widget_group(
            "Training split (%)",
            self.train_split_percent_choice.container,
            data_tab_l,
        )
        #######################
        #######################
        ui.add_blank(self, data_tab_l)
        #######################
        #######################
        data_tab_l.addWidget(
            ui.combine_blocks(
                left_or_above=self._make_prev_button(),
                right_or_below=self._make_next_button(),
                l=1,
            ),
            alignment=ui.LEFT_AL,
        )

        data_tab_l.addWidget(self.close_buttons[1], alignment=ui.LEFT_AL)
        ##################
        ############
        ######
        # Third tab : training parameters
        ######
        ############
        ##################
        train_tab = ui.ContainerWidget()
        ##################
        # ui.add_blank(train_tab, train_tab.layout)
        ##################
        self.loss_group = ui.GroupedWidget.create_single_widget_group(
            "Loss",
            self.loss_choice,
            train_tab.layout,
        )  # loss choice
        self.lbl_loss_choice.setVisible(False)
        # end of solo groups for loss
        ##################
        ui.add_blank(train_tab, train_tab.layout)
        ##################
        # training params group
        train_param_group_w, train_param_group_l = ui.make_group(
            "Training parameters", r=1, b=5, t=11
        )

        ui.add_widgets(
            train_param_group_l,
            [
                self.batch_choice.container,  # batch size
                self.lbl_learning_rate_choice,
                self.learning_rate_choice,
                self.epoch_choice.label,  # epochs
                self.epoch_choice,
                self.val_interval_choice.label,
                self.val_interval_choice,  # validation interval
                self.scheduler_patience_choice.label,
                self.scheduler_patience_choice,
                self.scheduler_factor_choice.label,
                self.scheduler_factor_choice.container,
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
        # self.container_seed.setVisible(False)
        self.use_deterministic_choice.setChecked(True)
        seed_w.setLayout(seed_l)
        train_tab.layout.addWidget(seed_w)
        # end of deterministic choice group
        ##################
        # buttons

        ui.add_blank(self, train_tab.layout)

        self.advanced_next_button = self._make_next_button()
        self.advanced_next_button.setVisible(False)
        self.start_button_supervised = self._make_start_button()

        ui.add_widgets(
            train_tab.layout,
            [
                ui.combine_blocks(
                    left_or_above=self._make_prev_button(),  # previous
                    right_or_below=self.advanced_next_button,  # next (only if unsupervised)
                    l=1,
                ),
                self.start_button_supervised,  # start
                ui.add_blank(self),
                self.close_buttons[2],
            ],
        )
        ##################
        ############
        ######
        # Fourth tab : advanced parameters (unsupervised only)
        ######
        ############
        ##################
        advanced_tab = ui.ContainerWidget(parent=self)
        self.wnet_widgets = WNetWidgets(parent=advanced_tab)
        ui.add_blank(advanced_tab, advanced_tab.layout)
        ##################
        model_params_group_w, model_params_group_l = ui.make_group(
            "WNet parameters", r=20, b=5, t=11
        )
        ui.add_widgets(
            model_params_group_l,
            [
                self.wnet_widgets.num_classes_choice.label,
                self.wnet_widgets.num_classes_choice,
                self.wnet_widgets.loss_choice.label,
                self.wnet_widgets.loss_choice,
            ],
        )
        model_params_group_w.setLayout(model_params_group_l)
        advanced_tab.layout.addWidget(model_params_group_w)
        ##################
        ui.add_blank(advanced_tab, advanced_tab.layout)
        ##################
        ncuts_loss_params_group_w, ncuts_loss_params_group_l = ui.make_group(
            "NCuts loss parameters", r=35, b=5, t=11
        )
        ui.add_widgets(
            ncuts_loss_params_group_l,
            [
                self.wnet_widgets.intensity_sigma_choice.label,
                self.wnet_widgets.intensity_sigma_choice,
                self.wnet_widgets.spatial_sigma_choice.label,
                self.wnet_widgets.spatial_sigma_choice,
                self.wnet_widgets.radius_choice.label,
                self.wnet_widgets.radius_choice,
            ],
        )
        ncuts_loss_params_group_w.setLayout(ncuts_loss_params_group_l)
        advanced_tab.layout.addWidget(ncuts_loss_params_group_w)
        ##################
        ui.add_blank(advanced_tab, advanced_tab.layout)
        ##################
        losses_weights_group_w, losses_weights_group_l = ui.make_group(
            "Losses weights", r=1, b=5, t=11
        )

        # container for reconstruction weight and divide factor
        reconstruction_weight_container = ui.ContainerWidget(
            vertical=False, parent=losses_weights_group_w
        )
        ui.add_widgets(
            reconstruction_weight_container.layout,
            [
                self.wnet_widgets.reconstruction_weight_choice,
                ui.make_label(" / "),
                self.wnet_widgets.reconstruction_weight_divide_factor_choice,
            ],
        )

        ui.add_widgets(
            losses_weights_group_l,
            [
                self.wnet_widgets.ncuts_weight_choice.label,
                self.wnet_widgets.ncuts_weight_choice,
                self.wnet_widgets.reconstruction_weight_choice.label,
                reconstruction_weight_container,
            ],
        )
        losses_weights_group_w.setLayout(losses_weights_group_l)
        advanced_tab.layout.addWidget(losses_weights_group_w)
        ##################
        ui.add_blank(advanced_tab, advanced_tab.layout)
        ##################
        # buttons
        self.start_button_unsupervised = self._make_start_button()
        ui.add_widgets(
            advanced_tab.layout,
            [
                self._make_prev_button(),  # previous
                self.start_button_unsupervised,  # start
                ui.add_blank(self),
                self.close_buttons[3],
            ],
        )
        ##################
        ############
        ######
        # end of tab layouts
        ui.ScrollArea.make_scrollable(
            contained_layout=model_tab.layout,
            parent=model_tab,
            min_wh=[200, 300],
        )  # , max_wh=[200,1000])

        ui.ScrollArea.make_scrollable(
            contained_layout=data_tab_l,
            parent=data_tab_w,
            min_wh=[200, 300],
        )

        ui.ScrollArea.make_scrollable(
            contained_layout=train_tab.layout,
            parent=train_tab,
            min_wh=[200, 300],
        )
        ui.ScrollArea.make_scrollable(
            contained_layout=advanced_tab.layout,
            parent=advanced_tab,
            min_wh=[200, 300],
        )

        self.addTab(model_tab, "Model")
        self.addTab(data_tab_w, "Data")
        self.addTab(train_tab, "Training")
        self.addTab(advanced_tab, "Advanced")
        self.setMinimumSize(220, 100)

        self._hide_unused()

        default_results_path = (
            config.SupervisedTrainingWorkerConfig().results_path_folder
        )
        self.results_filewidget.text_field.setText(default_results_path)
        self.results_filewidget.check_ready()
        self._check_results_path(default_results_path)
        self.results_path = default_results_path

    def send_log(self, text):
        """Sends a message via the Log attribute."""
        self.log.print_and_log(text)

    def start(self):
        """Initiates the :py:func:`train` function as a worker and does the following.

        * Checks that filepaths are set correctly using :py:func:`check_ready`

        * If self.worker is None : creates a worker and starts the training

        * If the button is clicked while training, stops the model once it finishes the next validation step and saves the results if better

        * When the worker yields after a validation step, plots the loss if epoch >= validation_step (to avoid empty plot on first validation)

        * When the worker finishes, clears the memory (tries to for now)

        Todo:
        * Fix memory allocation from torch


        Returns: Returns empty immediately if the file paths are not set correctly.

        """
        self.start_time = utils.get_time_filepath()

        if self._stop_requested:
            self.log.print_and_log("Worker is already stopping !")
            if self.worker is None:
                self._stop_requested = False
            return

        if not self.check_ready():  # issues a warning if not ready
            err = "Aborting, please set all required paths"
            # self.log.print_and_log(err)
            logger.warning(err)
            warnings.warn(err, stacklevel=1)
            return

        if self.worker is not None:
            if self.worker.is_running:
                pass
            else:
                self.worker.start()
                self.start_btn.setText("Running... Click to stop")
        else:  # starting a new job goes here
            self.log.print_and_log("Starting...")
            self.log.print_and_log("*" * 20)

            self._reset_loss_plot()

            self.config = config.TrainerConfig(
                save_as_zip=self.zip_choice.isChecked()
            )

            if self.unsupervised_mode:
                try:
                    self.data = self.create_dataset_dict_no_labs()
                except ValueError as err:
                    self.data = None
                    raise err
            else:
                try:
                    self.data = self.create_train_dataset_dict()
                except ValueError as err:
                    self.data = None
                    raise err

            # self._set_worker_config()
            self.worker = self._create_worker()  # calls _set_worker_config

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
            self._stop_requested = True
            self.start_btn.setText("Stopping... Please wait")
            self.log.print_and_log("*" * 20)
            self.worker.quit()
        else:
            self.worker.start()
            self.start_btn.setText("Running...  Click to stop")

    def _create_supervised_worker_from_config(
        self, worker_config: config.SupervisedTrainingWorkerConfig
    ):
        if isinstance(config, config.TrainerConfig):
            raise TypeError(
                "Expected a SupervisedTrainingWorkerConfig, got a TrainerConfig"
            )
        return SupervisedTrainingWorker(worker_config=worker_config)

    def _create_unsupervised_worker_from_config(
        self, worker_config: config.WNetTrainingWorkerConfig
    ):
        return WNetTrainingWorker(worker_config=worker_config)

    def _create_worker(self, additional_results_description=None):
        self._set_worker_config(
            additional_description=additional_results_description
        )
        if self.unsupervised_mode:
            return self._create_unsupervised_worker_from_config(
                self.worker_config
            )
        return self._create_supervised_worker_from_config(self.worker_config)

    def _set_worker_config(
        self,
        additional_description=None,
    ) -> config.TrainingWorkerConfig:
        """Creates a worker config for supervised or unsupervised training.

        Args:
            additional_description: Additional description to add to the results folder name.

        Returns:
            A worker config
        """
        logger.debug("Loading config...")
        model_config = config.ModelInfo(name=self.model_choice.currentText())

        self.weights_config.path = self.weights_config.path
        self.weights_config.use_custom = self.custom_weights_choice.isChecked()

        self.weights_config.use_pretrained = (
            self.use_transfer_choice.isChecked()
            and not self.custom_weights_choice.isChecked()
        )
        self.weights_config.use_custom = self.custom_weights_choice.isChecked()

        deterministic_config = config.DeterministicConfig(
            enabled=self.use_deterministic_choice.isChecked(),
            seed=self.box_seed.value(),
        )

        loss_name = (
            (f"{self.loss_choice.currentText()}_")
            if not self.unsupervised_mode
            else ""
        )
        additional_description = (
            (f"{additional_description}_")
            if additional_description is not None
            else ""
        )
        results_path_folder = Path(
            self.results_path
            + f"/{model_config.name}_"
            + additional_description
            + loss_name
            + f"{self.epoch_choice.value()}e_"
            + f"{utils.get_date_time()}"
        )
        Path(results_path_folder).mkdir(
            parents=True, exist_ok=False
        )  # avoid overwrite where possible
        patch_size = [w.value() for w in self.patch_size_widgets]

        if self.unsupervised_mode:
            try:
                self.unsupervised_eval_data = self.create_train_dataset_dict()
            except ValueError:
                self.unsupervised_eval_data = None
            self.worker_config = self._set_unsupervised_worker_config(
                results_path_folder,
                patch_size,
                deterministic_config,
                self.unsupervised_eval_data,
            )
        else:
            self.worker_config = self._set_supervised_worker_config(
                model_config,
                results_path_folder,
                patch_size,
                deterministic_config,
            )
        return self.worker_config

    def _set_supervised_worker_config(
        self,
        model_config,
        results_path_folder,
        patch_size,
        deterministic_config,
    ):
        """Sets the worker config for supervised training.

        Args:
            model_config: Model config
            results_path_folder: Path to results folder
            patch_size: Patch size
            deterministic_config: Deterministic config.

        Returns:
            A worker config
        """
        validation_percent = self.train_split_percent_choice.slider_value / 100
        self.worker_config = config.SupervisedTrainingWorkerConfig(
            device=self.check_device_choice(),
            model_info=model_config,
            weights_info=self.weights_config,
            train_data_dict=self.data,
            training_percent=validation_percent,
            max_epochs=self.epoch_choice.value(),
            loss_function=self.loss_choice.currentText(),
            learning_rate=self.learning_rate_choice.get_learning_rate(),
            scheduler_patience=self.scheduler_patience_choice.value(),
            scheduler_factor=self.scheduler_factor_choice.slider_value,
            validation_interval=self.val_interval_choice.value(),
            batch_size=self.batch_choice.slider_value,
            results_path_folder=str(results_path_folder),
            sampling=self.patch_choice.isChecked(),
            num_samples=self.sample_choice_slider.slider_value,
            sample_size=patch_size,
            do_augmentation=self.augment_choice.isChecked(),
            deterministic_config=deterministic_config,
        )

        return self.worker_config

    def _set_unsupervised_worker_config(
        self,
        results_path_folder,
        patch_size,
        deterministic_config,
        eval_volume_dict,
    ) -> config.WNetTrainingWorkerConfig:
        """Sets the worker config for unsupervised training.

        Args:
            results_path_folder: Path to results folder
            patch_size: Patch size
            deterministic_config: Deterministic config
            eval_volume_dict: Evaluation volume dictionary.

        Returns:
            A worker config
        """
        batch_size = self.batch_choice.slider_value
        if eval_volume_dict is None:
            eval_batch_size = 1
        else:
            eval_batch_size = (
                1 if len(eval_volume_dict) < batch_size else batch_size
            )
        self.worker_config = config.WNetTrainingWorkerConfig(
            device=self.check_device_choice(),
            weights_info=self.weights_config,
            train_data_dict=self.data,
            max_epochs=self.epoch_choice.value(),
            learning_rate=self.learning_rate_choice.get_learning_rate(),
            validation_interval=self.val_interval_choice.value(),
            batch_size=batch_size,
            results_path_folder=str(results_path_folder),
            sampling=self.patch_choice.isChecked(),
            num_samples=self.sample_choice_slider.slider_value,
            sample_size=patch_size,
            do_augmentation=self.augment_choice.isChecked(),
            deterministic_config=deterministic_config,
            num_classes=int(
                self.wnet_widgets.num_classes_choice.currentText()
            ),
            reconstruction_loss=self.wnet_widgets.loss_choice.currentText(),
            n_cuts_weight=self.wnet_widgets.ncuts_weight_choice.value(),
            rec_loss_weight=self.wnet_widgets.get_reconstruction_weight(),
            eval_volume_dict=eval_volume_dict,
            eval_batch_size=eval_batch_size,
        )

        return self.worker_config

    def _is_current_job_supervised(
        self,
    ):  # TODO(cyril) rework for better check and _make_csv
        if isinstance(self.worker, WNetTrainingWorker):
            return False
        return True

    def on_start(self):
        """Catches started signal from worker."""
        self.remove_docked_widgets()
        self.display_status_report()
        self.log.clear()
        self._remove_result_layers()

        self.log.print_and_log(f"Worker started at {utils.get_time()}")
        self.log.print_and_log("\nWorker is running...")

    def on_finish(self):
        """Catches finished signal from worker."""
        self.log.print_and_log("*" * 20)
        self.log.print_and_log(f"\nWorker finished at {utils.get_time()}")

        self.log.print_and_log(
            f"Saving in {self.worker_config.results_path_folder}"
        )
        self.log.print_and_log("Saving last loss plot")

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
        try:
            self._make_csv()
        except ValueError as e:
            logger.warning(f"Error while saving CSV report: {e}")

        self.start_btn.setText("Start")
        [btn.setVisible(True) for btn in self.close_buttons]

        if self.config.save_as_zip:
            shutil.make_archive(
                self.worker_config.results_path_folder,
                "zip",
                self.worker_config.results_path_folder,
            )
        self.worker = None

    def on_error(self):
        """Catches errored signal from worker."""
        self.log.print_and_log(f"WORKER ERRORED at {utils.get_time()}")
        self.worker = None

    def on_stop(self):
        """Catches stop signal from worker."""
        self._remove_result_layers()
        self.worker = None
        self._stop_requested = False
        self.start_btn.setText("Start")
        [btn.setVisible(True) for btn in self.close_buttons]

    def _remove_result_layers(self):
        for layer in self.result_layers:
            try:
                self._viewer.layers.remove(layer)
            except ValueError:
                logger.debug("Layer already removed ?")
                pass
        self.result_layers = []

    def _display_results(self, images_dict, complete_missing=False):
        """Show various model input/outputs in napari viewer as a list of layers."""
        layer_list = []
        if not complete_missing:
            for layer_name in list(images_dict.keys()):
                logger.debug(f"Adding layer {layer_name}")
                layer = self._viewer.add_image(
                    data=images_dict[layer_name]["data"],
                    name=layer_name,
                    colormap=images_dict[layer_name]["cmap"],
                )
                layer_list.append(layer)
            self.result_layers += layer_list
            self._viewer.grid.enabled = True
            self._viewer.dims.ndisplay = 3
            self._viewer.reset_view()
        else:
            for i, layer_name in enumerate(list(images_dict.keys())):
                if layer_name not in [
                    layer.name for layer in self._viewer.layers
                ]:
                    logger.debug(f"Adding missing layer {layer_name}")
                    layer = self._viewer.add_image(
                        data=images_dict[layer_name]["data"],
                        name=layer_name,
                        colormap=images_dict[layer_name]["cmap"],
                    )
                    layer_list[i] = layer
                else:
                    logger.debug(f"Refreshing layer {layer_name}")
                    self.result_layers[i].data = images_dict[layer_name][
                        "data"
                    ]
                    self.result_layers[i].refresh()
                    self.result_layers[i].reset_contrast_limits()

    def on_yield(self, report: TrainingReport):
        """Catches yielded signal from worker and plots the loss."""
        if report == TrainingReport():
            return  # skip empty reports

        if report.show_plot:
            try:
                self.log.print_and_log(len(report.images_dict))

                if (
                    report.epoch == 0
                    or report.epoch + 1
                    == self.worker_config.validation_interval
                ) and len(self.result_layers) == 0:
                    self.result_layers = []
                    self._display_results(report.images_dict)
                else:
                    self._display_results(
                        report.images_dict, complete_missing=True
                    )
            except Exception as e:
                logger.exception(e)

            self.progress.setValue(
                100 * (report.epoch + 1) // self.worker_config.max_epochs
            )

            self.update_loss_plot(report.loss_1_values, report.loss_2_values)
            self.loss_1_values = report.loss_1_values
            self.loss_2_values = report.loss_2_values

        if self._stop_requested:
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
            self.on_stop()
            self._stop_requested = False

    def _make_csv(self):
        size_column = range(1, self.worker_config.max_epochs + 1)

        if len(self.loss_1_values) == 0 or self.loss_1_values is None:
            logger.warning("No loss values to add to csv !")
            return

        try:
            self.loss_1_values["Loss"]
            supervised = True
        except KeyError("Loss"):
            try:
                self.loss_1_values["SoftNCuts"]
                supervised = False
            except KeyError("SoftNCuts") as e:
                raise KeyError(
                    "Error when making csv. Check loss dict keys ?"
                ) from e

        if supervised:
            val = utils.fill_list_in_between(
                self.loss_2_values,
                self.worker_config.validation_interval - 1,
                "",
            )[: len(size_column)]

            self.df = pd.DataFrame(
                {
                    "epoch": size_column,
                    "loss": self.loss_1_values["Loss"],
                    "validation": val,
                }
            )
            if len(val) != len(self.loss_1_values):
                err = f"Validation and loss values don't have the same length ! Got {len(val)} and {len(self.loss_1_values)}"
                logger.error(err)
                raise ValueError(err)
        else:
            ncuts_loss = self.loss_1_values["SoftNCuts"]
            try:
                dice_metric = self.loss_1_values["Dice metric"]
                self.df = pd.DataFrame(
                    {
                        "Epoch": size_column,
                        "Ncuts loss": ncuts_loss,
                        "Dice metric": dice_metric,
                        "Reconstruction loss": self.loss_2_values,
                    }
                )
            except KeyError:
                self.df = pd.DataFrame(
                    {
                        "Epoch": size_column,
                        "Ncuts loss": ncuts_loss,
                        "Reconstruction loss": self.loss_2_values,
                    }
                )

        path = Path(self.worker_config.results_path_folder) / Path(
            "training.csv"
        )
        self.df.to_csv(path, index=False)

    def _show_plot_max(self, plot, y):
        x_max = (np.argmax(y) + 1) * self.worker_config.validation_interval
        dice_max = np.max(y)
        plot.scatter(
            x_max,
            dice_max,
            c="r",
            label="Max. Dice",
            zorder=5,
        )

    def _plot_loss(
        self,
        loss_values_1: dict,
        loss_values_2: list,
        show_plot_2_max: bool = True,
    ):
        """Creates two subplots to plot the training loss and validation metric."""
        plot_key = (
            "supervised"
            if self._is_current_job_supervised()
            else "unsupervised"
        )
        with plt.style.context("dark_background"):
            # update loss
            self.plot_1.set_title(self.plot_1_labels["title"][plot_key])
            self.plot_1.set_xlabel("Epoch")
            self.plot_1.set_ylabel(self.plot_2_labels["ylabel"][plot_key])

            for metric_name in list(loss_values_1.keys()):
                if metric_name == "Dice metric":
                    x = [
                        self.worker_config.validation_interval * (i + 1)
                        for i in range(len(loss_values_1[metric_name]))
                    ]
                else:
                    x = [i + 1 for i in range(len(loss_values_1[metric_name]))]
                y = loss_values_1[metric_name]
                self.plot_1.plot(x, y, label=metric_name)
                if metric_name == "Dice metric":
                    self._show_plot_max(self.plot_1, y)
            if len(loss_values_1.keys()) > 1:
                self.plot_1.legend(
                    loc="lower left", fontsize="10", markerscale=0.6
                )

            # update plot 2
            if self._is_current_job_supervised():
                x = [
                    int(self.worker_config.validation_interval * (i + 1))
                    for i in range(len(loss_values_2))
                ]
            else:
                x = [int(i + 1) for i in range(len(loss_values_2))]
            y = loss_values_2

            self.plot_2.plot(x, y, zorder=1)
            # self.dice_metric_plot.set_ylim(0, 1)
            self.plot_2.set_title(self.plot_2_labels["title"][plot_key])
            self.plot_2.set_xlabel("Epoch")
            self.plot_2.set_ylabel(self.plot_2_labels["ylabel"][plot_key])

            if show_plot_2_max:
                self._show_plot_max(self.plot_2, y)
                self.plot_2.legend(facecolor=ui.napari_grey, loc="lower right")
            self.canvas.draw_idle()

    def update_loss_plot(self, loss_1: dict, loss_2: list):
        """Updates the plots on subsequent validation steps.

        Creates the plot on the second validation step (epoch == val_interval*2).
        Updates the plot on subsequent validation steps.
        Epoch is obtained from the length of the loss vector.

        Returns: returns empty if the epoch is < than 2 * validation interval.
        """
        epoch = len(loss_1[list(loss_1.keys())[0]])
        logger.debug(f"Updating loss plot for epoch {epoch}")
        plot_max = self._is_current_job_supervised()
        if epoch < self.worker_config.validation_interval * 2:
            return
        if epoch == self.worker_config.validation_interval * 2:
            bckgrd_color = (0, 0, 0, 0)  # '#262930'
            with plt.style.context("dark_background"):
                self.canvas = FigureCanvas(Figure(figsize=(10, 1.5)))
                # loss plot
                self.plot_1 = self.canvas.figure.add_subplot(1, 2, 1)
                # dice metric validation plot
                self.plot_2 = self.canvas.figure.add_subplot(1, 2, 2)

                self.canvas.figure.set_facecolor(bckgrd_color)
                self.plot_2.set_facecolor(bckgrd_color)
                self.plot_1.set_facecolor(bckgrd_color)

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
                self.docked_widgets.append(self.plot_dock)
            except AttributeError as e:
                logger.exception(e)
                logger.error(
                    "Plot dock widget could not be added. Should occur in testing only"
                )
            self._plot_loss(loss_1, loss_2, show_plot_2_max=plot_max)
        else:
            with plt.style.context("dark_background"):
                self.plot_1.cla()
                self.plot_2.cla()
                self._plot_loss(loss_1, loss_2, show_plot_2_max=plot_max)

    def _reset_loss_plot(self):
        if self.plot_1 is not None and self.plot_2 is not None:
            with plt.style.context("dark_background"):
                self.plot_1.cla()
                self.plot_2.cla()


class LearningRateWidget(ui.ContainerWidget):
    """A widget to choose the learning rate."""

    def __init__(self, parent=None):
        """Creates a widget to choose the learning rate."""
        super().__init__(vertical=False, parent=parent)

        self.lr_exponent_dict = {
            "1e-2": 1e-2,
            "1e-3": 1e-3,
            "1e-4": 1e-4,
            "1e-5": 1e-5,
            "1e-6": 1e-6,
            "1e-7": 1e-7,
            "1e-8": 1e-8,
        }

        self.lr_value_choice = ui.IntIncrementCounter(
            lower=1,
            upper=9,
            default=1,
            text_label="Learning rate : ",
            parent=self,
            fixed=False,
        )
        self.lr_exponent_choice = ui.DropdownMenu(
            list(self.lr_exponent_dict.keys()),
            parent=self,
            fixed=False,
        )
        self._build()

    def _build(self):
        self.lr_value_choice.setFixedWidth(20)
        # self.lr_exponent_choice.setFixedWidth(100)
        self.lr_exponent_choice.setCurrentIndex(1)
        ui.add_widgets(
            self.layout,
            [
                self.lr_value_choice,
                ui.make_label("x"),
                self.lr_exponent_choice,
            ],
        )

    def get_learning_rate(self) -> float:
        """Return the learning rate as a float."""
        return float(
            self.lr_value_choice.value()
            * self.lr_exponent_dict[self.lr_exponent_choice.currentText()]
        )


class WNetWidgets:
    """A collection of widgets for the WNet training GUI."""

    default_config = config.WNetTrainingWorkerConfig()

    def __init__(self, parent):
        """Creates a collection of widgets for the WNet training GUI."""
        self.num_classes_choice = ui.DropdownMenu(
            entries=["2", "3", "4"],
            parent=parent,
            text_label="Number of classes",
        )
        self.intensity_sigma_choice = ui.DoubleIncrementCounter(
            lower=1.0,
            upper=100.0,
            default=self.default_config.intensity_sigma,
            parent=parent,
            text_label="Intensity sigma",
        )
        self.intensity_sigma_choice.setMaximumWidth(20)
        self.spatial_sigma_choice = ui.DoubleIncrementCounter(
            lower=1.0,
            upper=100.0,
            default=self.default_config.spatial_sigma,
            parent=parent,
            text_label="Spatial sigma",
        )
        self.spatial_sigma_choice.setMaximumWidth(20)
        self.radius_choice = ui.IntIncrementCounter(
            lower=1,
            upper=5,
            default=self.default_config.radius,
            parent=parent,
            text_label="Radius",
        )
        self.radius_choice.setMaximumWidth(20)
        self.loss_choice = ui.DropdownMenu(
            entries=["MSE", "BCE"],
            parent=parent,
            text_label="Reconstruction loss",
        )
        self.ncuts_weight_choice = ui.DoubleIncrementCounter(
            lower=0.01,
            upper=1.0,
            default=self.default_config.n_cuts_weight,
            parent=parent,
            text_label="NCuts weight",
        )
        self.reconstruction_weight_choice = ui.DoubleIncrementCounter(
            lower=0.01,
            upper=1.0,
            default=0.5,
            parent=parent,
            text_label="Reconstruction weight",
        )
        self.reconstruction_weight_choice.setMaximumWidth(20)
        self.reconstruction_weight_divide_factor_choice = (
            ui.IntIncrementCounter(
                lower=1,
                upper=10000,
                default=100,
                parent=parent,
                text_label="Reconstruction weight divide factor",
            )
        )
        self.reconstruction_weight_divide_factor_choice.setMaximumWidth(20)

        self._set_tooltips()

    def _set_tooltips(self):
        self.num_classes_choice.setToolTip("Number of classes to segment")
        self.intensity_sigma_choice.setToolTip(
            "Intensity sigma for the NCuts loss"
        )
        self.spatial_sigma_choice.setToolTip(
            "Spatial sigma for the NCuts loss"
        )
        self.radius_choice.setToolTip("Radius of NCuts loss region")
        self.loss_choice.setToolTip("Loss function to use for reconstruction")
        self.ncuts_weight_choice.setToolTip("Weight of the NCuts loss")
        self.reconstruction_weight_choice.setToolTip(
            "Weight of the reconstruction loss"
        )
        self.reconstruction_weight_divide_factor_choice.setToolTip(
            "Divide factor for the reconstruction loss.\nThis might have to be changed depending on your images.\nIf you notice that the reconstruction loss is too high, raise this factor until the\nreconstruction loss is in the same order of magnitude as the NCuts loss."
        )

    def get_reconstruction_weight(self):
        """Returns the reconstruction weight as a float."""
        return float(
            self.reconstruction_weight_choice.value()
            / self.reconstruction_weight_divide_factor_choice.value()
        )
