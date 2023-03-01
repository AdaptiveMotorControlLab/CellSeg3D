import warnings
from functools import partial

import napari
import numpy as np
import pandas as pd

# local
from napari_cellseg3d import config
from napari_cellseg3d import interface as ui
from napari_cellseg3d import utils
from napari_cellseg3d.code_models.model_framework import ModelFramework
from napari_cellseg3d.code_models.model_workers import InferenceResult
from napari_cellseg3d.code_models.model_workers import InferenceWorker
from napari_cellseg3d.code_plugins.plugin_convert import InstanceWidgets


class Inferer(ModelFramework, metaclass=ui.QWidgetSingleton):
    """A plugin to run already trained models in evaluation mode to preform inference and output a label on all
    given volumes."""

    def __init__(self, viewer: "napari.viewer.Viewer", parent=None):
        """
        Creates an Inference loader plugin with the following widgets :

        * Data :
            * A file extension choice for the images to load from selected folders

            * Two fields to choose the images folder to run segmentation and save results in, respectively

        * Inference options :
            * A dropdown menu to select which model should be used for inference

            * An option to load custom weights for the selected model (e.g. from training module)


        * Post-processing :
            * A box to select if data is anisotropic, if checked, asks for resolution in micron for each axis

            * A box to choose whether to threshold, if checked asks for a threshold between 0 and 1

            * A box to enable instance segmentation. If enabled, displays :
                * The choice of method to use for instance segmentation

                * The probability threshold below which to remove objects

                * The size in pixels of small objects to remove

        * A checkbox to choose whether to display results in napari afterwards. Will ask for how many results to display, capped at 10

        * A button to launch the inference process

        * A button to close the widget

        Args:
            viewer (napari.viewer.Viewer): napari viewer to display the widget in
        """
        super().__init__(
            viewer,
            parent,
            loads_labels=False,
        )

        self._viewer = viewer
        """Viewer to display the widget in"""
        self.enable_utils_menu()

        self.worker: InferenceWorker = None
        """Worker for inference, should be an InferenceWorker instance from model_workers.py"""

        self.model_info: config.ModelInfo = None
        """ModelInfo class from config.py"""

        self.config = config.InfererConfig()
        """InfererConfig class from config.py"""
        self.worker_config: config.InferenceWorkerConfig = (
            config.InferenceWorkerConfig()
        )
        """InferenceWorkerConfig class from config.py"""
        self.instance_config: config.InstanceSegConfig = (
            config.InstanceSegConfig()
        )
        """InstanceSegConfig class from config.py"""
        self.post_process_config: config.PostProcessConfig = (
            config.PostProcessConfig()
        )
        """PostProcessConfig class from config.py"""

        ###########################
        # interface
        self.data_panel = self._build_io_panel()  # None

        self.view_results_container = ui.ContainerWidget(t=7, b=0, parent=self)
        self.view_results_panel = None

        self.view_checkbox = ui.CheckBox(
            "View results in napari", self._toggle_display_number
        )

        self.display_number_choice_slider = ui.Slider(
            lower=1, upper=10, default=5, text_label="How many ? "
        )

        self.show_original_checkbox = ui.CheckBox("Show originals")

        ######################
        ######################
        # TODO : better way to handle SegResNet size reqs ?
        self.model_input_size = ui.IntIncrementCounter(
            lower=1, upper=1024, default=128, label="\nModel input size"
        )
        self.model_choice.currentIndexChanged.connect(
            self._toggle_display_model_input_size
        )
        self.model_choice.setCurrentIndex(0)

        self.anisotropy_wdgt = ui.AnisotropyWidgets(
            self,
            default_x=1.5,
            default_y=1.5,
            default_z=5,  # TODO change default
        )

        # self.worker_config.post_process_config.zoom.zoom_values = [
        #     1.0,
        #     1.0,
        #     1.0,
        # ]

        # ui.add_blank(self.aniso_container, aniso_layout)

        ######################
        ######################
        self.thresholding_checkbox = ui.CheckBox(
            "Perform thresholding", self._toggle_display_thresh
        )

        self.thresholding_slider = ui.Slider(
            lower=1,
            default=config.PostProcessConfig().thresholding.threshold_value
            * 100,
            divide_factor=100.0,
            parent=self,
        )

        self.window_infer_box = ui.CheckBox("Use window inference")
        self.window_infer_box.clicked.connect(self._toggle_display_window_size)

        sizes_window = ["8", "16", "32", "64", "128", "256", "512"]
        # (
        #     self.window_size_choice,
        #     self.window_size_choice.label,
        # ) = ui.make_combobox(sizes_window, label="Window size and overlap")
        # self.window_overlap = ui.make_n_spinboxes(
        #     max=1,
        #     default=0.7,
        #     step=0.05,
        #     double=True,
        # )

        self.window_size_choice = ui.DropdownMenu(
            sizes_window, label="Window size"
        )

        self.window_overlap_slider = ui.Slider(
            default=config.SlidingWindowConfig.window_overlap * 100,
            divide_factor=100.0,
            parent=self,
            text_label="Overlap %",
        )

        self.keep_data_on_cpu_box = ui.CheckBox("Keep data on CPU")

        window_size_widgets = ui.combine_blocks(
            self.window_size_choice,
            self.window_size_choice.label,
            horizontal=False,
        )

        self.window_infer_params = ui.ContainerWidget(parent=self)
        ui.add_widgets(
            self.window_infer_params.layout,
            [
                window_size_widgets,
                self.window_overlap_slider.container,
            ],
        )

        ##################
        ##################
        # instance segmentation widgets
        self.instance_widgets = InstanceWidgets(self)

        self.use_instance_choice = ui.CheckBox(
            "Run instance segmentation", func=self._toggle_display_instance
        )

        self.save_stats_to_csv_box = ui.CheckBox(
            "Save stats to csv", parent=self
        )

        ##################
        ##################

        self.btn_start = ui.Button("Start", self.start)
        self.btn_close = self._make_close_button()

        self._set_tooltips()

        self._build()
        self._set_io_visibility()
        self.folder_choice.toggled.connect(
            partial(
                self._show_io_element,
                self.view_results_panel,
                self.folder_choice,
            )
        )
        self.folder_choice.toggle()
        self.layer_choice.toggle()

        self._remove_unused()

    def _set_tooltips(self):
        ##################
        ##################
        # tooltips
        self.view_checkbox.setToolTip("Show results in the napari viewer")
        self.display_number_choice_slider.tooltips = (
            "Choose how many results to display once the work is done.\n"
            "Maximum is 10 for clarity"
        )
        self.show_original_checkbox.setToolTip(
            "Displays the image used for inference in the viewer"
        )
        self.model_input_size.setToolTip(
            "Image size on which the model has been trained (default : 128)\n"
            "DO NOT CHANGE if you are using the provided pre-trained weights"
        )

        thresh_desc = (
            "Thresholding : all values in the image below the chosen probability"
            " threshold will be set to 0, and all others to 1."
        )

        self.thresholding_checkbox.setToolTip(thresh_desc)
        self.thresholding_slider.tooltips = thresh_desc
        self.window_infer_box.setToolTip(
            "Sliding window inference runs the model on parts of the image"
            "\nrather than the whole image, to reduce memory requirements."
            "\nUse this if you have large images."
        )
        self.window_size_choice.setToolTip(
            "Size of the window to run inference with (in pixels)"
        )
        self.window_overlap_slider.tooltips = "Percentage of overlap between windows to use when using sliding window"

        self.keep_data_on_cpu_box.setToolTip(
            "If enabled, data will be kept on the RAM rather than the VRAM.\nCan avoid out of memory issues with CUDA"
        )
        self.use_instance_choice.setToolTip(
            "Instance segmentation will convert instance (0/1) labels to labels"
            " that attempt to assign an unique ID to each cell."
        )

        self.save_stats_to_csv_box.setToolTip(
            "Will save several statistics for each object to a csv in the results folder. Stats include : "
            "volume, centroid coordinates, sphericity"
        )
        ##################
        ##################

    def check_ready(self):
        """Checks if the paths to the files are properly set"""

        if self.layer_choice.isChecked():
            if self.image_layer_loader.layer_data() is not None:
                return True
        elif self.folder_choice.isChecked():
            if self.image_filewidget.check_ready():
                return True
        return False

    def _toggle_display_model_input_size(self):
        if (
            self.model_choice.currentText() == "SegResNet"
            or self.model_choice.currentText() == "SwinUNetR"
        ):
            self.model_input_size.setVisible(True)
            self.model_input_size.label.setVisible(True)
        else:
            self.model_input_size.setVisible(False)
            self.model_input_size.label.setVisible(False)

    def _toggle_display_number(self):
        """Shows the choices for viewing results depending on whether :py:attr:`self.view_checkbox` is checked"""
        ui.toggle_visibility(self.view_checkbox, self.view_results_container)

    def _toggle_display_thresh(self):
        """Shows the choices for thresholding results depending on whether :py:attr:`self.thresholding_checkbox` is checked"""
        ui.toggle_visibility(
            self.thresholding_checkbox, self.thresholding_slider.container
        )

    def _toggle_display_instance(self):
        """Shows or hides the options for instance segmentation based on current user selection"""
        ui.toggle_visibility(self.use_instance_choice, self.instance_widgets)

    def _toggle_display_window_size(self):
        """Show or hide window size choice depending on status of self.window_infer_box"""
        ui.toggle_visibility(self.window_infer_box, self.window_infer_params)

    def _build(self):
        """Puts all widgets in a layout and adds them to the napari Viewer"""

        # ui.add_blank(self.view_results_container, view_results_layout)
        ui.add_widgets(
            self.view_results_container.layout,
            [
                self.view_checkbox,
                self.display_number_choice_slider.container,
                self.show_original_checkbox,
            ],
            alignment=None,
        )

        self.view_results_container.setLayout(
            self.view_results_container.layout
        )

        self.anisotropy_wdgt.build()

        ######
        ############
        ##################
        tab = ui.ContainerWidget(
            b=1, parent=self
        )  # tab that will contain all widgets

        L, T, R, B = 7, 20, 7, 11  # margins for group boxes
        #################################
        #################################
        # self.image_filewidget.update_field_color("black")

        self.results_filewidget.text_field.setText(
            self.worker_config.results_path
        )
        self.results_filewidget.check_ready()

        tab.layout.addWidget(self.data_panel)
        #################################
        #################################
        ui.add_blank(tab, tab.layout)
        #################################
        #################################
        # model group

        model_group_w, model_group_l = ui.make_group(
            "Model choice", L, T, R, B, parent=self
        )  # model choice

        ui.add_widgets(
            model_group_l,
            [
                self.model_choice,
                self.custom_weights_choice,
                self.weights_filewidget,
                self.model_input_size.label,
                self.model_input_size,
            ],
        )
        self.weights_filewidget.setVisible(False)
        self.model_choice.label.setVisible(
            False
        )  # TODO reminder for adding custom model

        model_group_w.setLayout(model_group_l)
        tab.layout.addWidget(model_group_w)
        #################################
        #################################
        ui.add_blank(tab, tab.layout)
        #################################
        #################################
        inference_param_group_w, inference_param_group_l = ui.make_group(
            "Inference parameters", parent=self
        )

        ui.add_widgets(
            inference_param_group_l,
            [
                self.window_infer_box,
                self.window_infer_params,
                self.keep_data_on_cpu_box,
            ],
        )
        self.window_infer_params.setVisible(False)

        inference_param_group_w.setLayout(inference_param_group_l)

        tab.layout.addWidget(inference_param_group_w)

        #################################
        #################################
        ui.add_blank(tab, tab.layout)
        #################################
        #################################
        # post proc group
        post_proc_group, post_proc_layout = ui.make_group(
            "Post-processing", parent=self
        )

        self.thresholding_slider.container.setVisible(False)

        ui.add_widgets(
            post_proc_layout,
            [
                self.anisotropy_wdgt,  # anisotropy
                self.thresholding_checkbox,
                self.thresholding_slider.container,  # thresholding
                self.use_instance_choice,
                self.instance_widgets,
                self.save_stats_to_csv_box,
                # self.instance_param_container,  # instance segmentation
            ],
        )
        ModelFramework._show_io_element(
            self.save_stats_to_csv_box, self.use_instance_choice
        )

        self.anisotropy_wdgt.container.setVisible(False)
        self.thresholding_slider.container.setVisible(False)
        self.instance_widgets.setVisible(False)
        self.save_stats_to_csv_box.setVisible(False)

        post_proc_group.setLayout(post_proc_layout)
        tab.layout.addWidget(post_proc_group, alignment=ui.LEFT_AL)
        ###################################
        ###################################
        ui.add_blank(tab, tab.layout)
        ###################################
        ###################################
        display_opt_group, display_opt_layout = ui.make_group(
            "Display options", L, T, R, B, parent=self
        )

        ui.add_widgets(
            display_opt_layout,
            [
                self.view_checkbox,  # ui.combine_blocks(self.view_checkbox, self.lbl_view),
                self.view_results_container,  # view_after bool
            ],
        )

        self.show_original_checkbox.toggle()
        self.view_results_container.setVisible(False)

        self.view_checkbox.toggle()
        self._toggle_display_number()

        # TODO : add custom model handling ?
        # self.label_filewidget.text_field.setText("model.pth directory :")

        display_opt_group.setLayout(display_opt_layout)
        self.view_results_panel = display_opt_group
        tab.layout.addWidget(display_opt_group)
        ###################################
        ui.add_blank(self, tab.layout)
        ###################################
        ###################################
        ui.add_widgets(
            tab.layout,
            [
                self.btn_start,
                self.btn_close,
            ],
        )
        ##################
        ############
        ######
        # end of tabs, combine into scrollable
        ui.ScrollArea.make_scrollable(
            parent=tab,
            contained_layout=tab.layout,
            min_wh=[200, 100],
        )
        self.addTab(tab, "Inference")
        tab.adjustSize()
        # self.setMinimumSize(180, 100)
        # self.setBaseSize(210, 400)

    def start(self):
        """Start the inference process, enables :py:attr:`~self.worker` and does the following:

        * Checks if the output and input folders are correctly set

        * Loads the weights from the chosen model

        * Creates a dict with all image paths (see :py:func:`~create_inference_dict`)

        * Loads the images, pads them so their size is a power of two in every dim (see :py:func:`utils.get_padding_dim`)

        * Performs sliding window inference (from MONAI) on every image, with or without ROI of chosen size

        * Saves all outputs in the selected results folder

        * If the option has been selected, display the results in napari, up to the maximum number selected

        * Runs instance segmentation, thresholding, and stats computing if requested

        Args:
            on_layer: if True, will start inference on a selected layer
        """

        if not self.check_ready():
            err = "Aborting, please choose valid inputs"
            self.log.print_and_log(err)
            raise ValueError(err)

        if self.worker is not None:
            if self.worker.is_running:
                pass
            else:
                self.worker.start()
                self.btn_start.setText("Running... Click to stop")
        else:
            self.log.print_and_log("Starting...")
            self.log.print_and_log("*" * 20)

            self.model_info = config.ModelInfo(
                name=self.model_choice.currentText(),
                model_input_size=self.model_input_size.value(),
            )

            self.weights_config.custom = self.custom_weights_choice.isChecked()

            save_path = self.results_filewidget.text_field.text()
            if not self._check_results_path(save_path):
                msg = f"ERROR: please set valid results path. Current path is {save_path}"
                self.log.print_and_log(msg)
                warnings.warn(msg)
            else:
                if self.results_path is None:
                    self.results_path = save_path

            zoom_config = config.Zoom(
                enabled=self.anisotropy_wdgt.enabled(),
                zoom_values=self.anisotropy_wdgt.scaling_xyz(),
            )
            thresholding_config = config.Thresholding(
                enabled=self.thresholding_checkbox.isChecked(),
                threshold_value=self.thresholding_slider.slider_value,
            )

            instance_thresh_config = config.Thresholding(
                threshold_value=self.instance_widgets.threshold_slider1.slider_value
            )
            instance_small_object_thresh_config = config.Thresholding(
                threshold_value=self.instance_widgets.counter1.value()
            )
            self.instance_config = config.InstanceSegConfig(
                enabled=self.use_instance_choice.isChecked(),
                method=self.instance_widgets.method_choice.currentText(),
                threshold=instance_thresh_config,
                small_object_removal_threshold=instance_small_object_thresh_config,
            )

            self.post_process_config = config.PostProcessConfig(
                zoom=zoom_config,
                thresholding=thresholding_config,
                instance=self.instance_config,
            )

            if self.window_infer_box.isChecked():
                size = int(self.window_size_choice.currentText())
                window_config = config.SlidingWindowConfig(
                    window_size=size,
                    window_overlap=self.window_overlap_slider.slider_value,
                )
            else:
                window_config = config.SlidingWindowConfig()

            self.worker_config = config.InferenceWorkerConfig(
                device=self.get_device(),
                model_info=self.model_info,
                weights_config=self.weights_config,
                results_path=self.results_path,
                filetype=self.filetype_choice.currentText(),
                keep_on_cpu=self.keep_data_on_cpu_box.isChecked(),
                compute_stats=self.save_stats_to_csv_box.isChecked(),
                post_process_config=self.post_process_config,
                sliding_window_config=window_config,
            )
            #####################
            #####################
            #####################

            if self.folder_choice.isChecked():

                self.worker_config.images_filepaths = self.images_filepaths
                self.worker = InferenceWorker(worker_config=self.worker_config)

            elif self.layer_choice.isChecked():

                self.worker_config.layer = self.image_layer_loader.layer_data()
                self.worker = InferenceWorker(worker_config=self.worker_config)

            else:
                raise ValueError("Please select to load a layer or folder")

            self.worker.set_download_log(self.log)

            self.worker.started.connect(self.on_start)
            self.worker.log_signal.connect(self.log.print_and_log)
            self.worker.warn_signal.connect(self.log.warn)
            self.worker.yielded.connect(partial(self.on_yield))  #
            self.worker.errored.connect(partial(self.on_yield))
            self.worker.finished.connect(self.on_finish)

            if self.get_device(show=False) == "cuda":
                self.worker.finished.connect(self.empty_cuda_cache)
            self.btn_close.setVisible(False)

        if self.worker.is_running:  # if worker is running, tries to stop
            self.log.print_and_log(
                "Stop request, waiting for next inference & saving to occur..."
            )
            self.btn_start.setText("Stopping...")
            self.worker.quit()
        else:  # once worker is started, update buttons
            self.worker.start()
            self.btn_start.setText("Running...  Click to stop")

    def on_start(self):
        """Catches start signal from worker to call :py:func:`~display_status_report`"""
        self.display_status_report()

        self.config = config.InfererConfig(
            model_info=self.model_info,
            show_results=self.view_checkbox.isChecked(),
            show_results_count=self.display_number_choice_slider.slider_value,
            show_original=self.show_original_checkbox.isChecked(),
            anisotropy_resolution=self.anisotropy_wdgt.resolution_xyz,
        )
        if self.layer_choice.isChecked():
            self.config.show_results = True
            self.config.show_results_count = 5
            self.config.show_original = False

        self.log.print_and_log(f"Worker started at {utils.get_time()}")
        self.log.print_and_log(f"Saving results to : {self.results_path}")
        self.log.print_and_log("Worker is running...")

    def on_error(self):
        """Catches errors and tries to clean up. TODO : upgrade"""
        self.log.print_and_log("Worker errored...")
        self.log.print_and_log("Trying to clean up...")
        self.btn_start.setText("Start")
        self.btn_close.setVisible(True)

        self.worker = None
        self.worker_config = None
        self.empty_cuda_cache()

    def on_finish(self):
        """Catches finished signal from worker, resets workspace for next run."""
        self.log.print_and_log(f"\nWorker finished at {utils.get_time()}")
        self.log.print_and_log("*" * 20)
        self.btn_start.setText("Start")
        self.btn_close.setVisible(True)

        self.worker = None
        self.worker_config = None
        self.empty_cuda_cache()

    def on_yield(self, result: InferenceResult):
        """
        Displays the inference results in napari as long as data["image_id"] is lower than nbr_to_show,
        and updates the status report docked widget (namely the progress bar)

        Args:
            data (dict): dict yielded by :py:func:`~inference()`, contains : "image_id" : index of the returned image, "original" : original volume used for inference, "result" : inference result
            widget (QWidget): widget for accessing attributes
        """
        # viewer, progress, show_res, show_res_number, zoon, show_original

        # check that viewer checkbox is on and that max number of displays has not been reached.
        # widget.log.print_and_log(result)

        image_id = result.image_id
        model_name = result.model_name
        if self.worker_config.images_filepaths is not None:
            total = len(self.worker_config.images_filepaths)
        else:
            total = 1

        viewer = self._viewer

        pbar_value = image_id // total
        if pbar_value == 0:
            pbar_value = 1

        self.progress.setValue(100 * pbar_value)

        if (
            self.config.show_results
            and image_id <= self.config.show_results_count
        ):

            zoom = self.worker_config.post_process_config.zoom.zoom_values

            viewer.dims.ndisplay = 3
            viewer.scale_bar.visible = True

            if self.config.show_original and result.original is not None:
                original_layer = viewer.add_image(
                    result.original,
                    colormap="inferno",
                    name=f"original_{image_id}",
                    scale=zoom,
                    opacity=0.7,
                )

            out_colormap = "twilight"
            if self.worker_config.post_process_config.thresholding.enabled:
                out_colormap = "turbo"

            out_layer = viewer.add_image(
                result.result,
                colormap=out_colormap,
                name=f"pred_{image_id}_{model_name}",
                opacity=0.8,
            )

            if result.instance_labels is not None:

                labels = result.instance_labels
                method = self.worker_config.post_process_config.instance.method

                number_cells = (
                    np.unique(labels.flatten()).size - 1
                )  # remove background

                name = f"({number_cells} objects)_{method}_instance_labels_{image_id}"

                instance_layer = viewer.add_labels(labels, name=name)

                stats = result.stats

                if self.worker_config.compute_stats and stats is not None:

                    stats_dict = stats.get_dict()
                    stats_df = pd.DataFrame(stats_dict)

                    self.log.print_and_log(
                        f"Number of instances : {stats.number_objects}"
                    )

                    csv_name = f"/{method}_seg_results_{image_id}_{utils.get_date_time()}.csv"
                    stats_df.to_csv(
                        self.worker_config.results_path + csv_name,
                        index=False,
                    )

                    # self.log.print_and_log(
                    #     f"OBJECTS DETECTED : {number_cells}\n"
                    # )
