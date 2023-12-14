"""Inference plugin for napari_cellseg3d."""
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import napari

# local
from napari_cellseg3d import config, utils
from napari_cellseg3d import interface as ui
from napari_cellseg3d.code_models.instance_segmentation import (
    InstanceMethod,
    InstanceWidgets,
)
from napari_cellseg3d.code_models.model_framework import ModelFramework
from napari_cellseg3d.code_models.worker_inference import InferenceWorker
from napari_cellseg3d.code_models.workers_utils import InferenceResult
from napari_cellseg3d.code_plugins.plugin_crf import CRFParamsWidget

logger = utils.LOGGER


class Inferer(ModelFramework, metaclass=ui.QWidgetSingleton):
    """A plugin to run already trained models in evaluation mode to preform inference and output a label on all given volumes."""

    def __init__(self, viewer: "napari.viewer.Viewer", parent=None):
        """Creates an Inference loader plugin with the following widgets.

        * Data :
            * A file extension choice for the images to load from selected folders

            * Two fields to choose the images folder to run segmentation and save results in, respectively

        * Inference options :
            * A dropdown menu to select which model should be used for inference

            * An option to load custom weights for the selected model (e.g. from training module)


        * Additional options :
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
            parent (QWidget, optional): Defaults to None.
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
        self.instance_config: InstanceMethod
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
            lower=1, upper=1024, default=64, text_label="\nModel input size"
        )
        self.model_choice.currentIndexChanged.connect(
            self._toggle_display_model_input_size
        )
        self.model_choice.currentIndexChanged.connect(
            self._restrict_window_size_for_model
        )
        self.model_choice.setCurrentIndex(0)

        self.anisotropy_wdgt = ui.AnisotropyWidgets(
            self,
            default_x=1,
            default_y=1,
            default_z=1,
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
            default=config.PostProcessConfig().thresholding.threshold_value
            * 100,
            divide_factor=100.0,
            parent=self,
        )

        self.use_window_choice = ui.CheckBox("Use window inference")
        self.use_window_choice.toggled.connect(
            self._toggle_display_window_size
        )

        sizes_window = ["8", "16", "32", "64", "128", "256", "512"]
        self._default_window_size = sizes_window.index("64")
        self.wnet_enabled = False

        self.window_size_choice = ui.DropdownMenu(
            sizes_window, text_label="Window size"
        )
        self.window_size_choice.setCurrentIndex(
            self._default_window_size
        )  # set to 64 by default

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
        # auto-artifact removal widgets
        self.attempt_artifact_removal_box = ui.CheckBox(
            "Attempt artifact removal",
            func=self._toggle_display_artifact_size_thresh,
            parent=self,
        )
        self.remove_artifacts_label = ui.make_label(
            "Remove labels larger than :"
        )
        self.artifact_removal_size = ui.IntIncrementCounter(
            lower=1,
            upper=10000,
            default=500,
            text_label="Remove larger than :",
            step=100,
        )
        self.artifact_container = ui.ContainerWidget(parent=self)
        self.attempt_artifact_removal_box.toggled.connect(
            self._toggle_display_artifact_size_thresh
        )
        ##################
        ##################
        # instance segmentation widgets
        self.instance_widgets = InstanceWidgets(parent=self)
        self.crf_widgets = CRFParamsWidget(parent=self)

        self.use_instance_choice = ui.CheckBox(
            "Run instance segmentation",
            func=self._toggle_display_instance,
            parent=self,
        )
        self.use_instance_choice.toggled.connect(
            self._toggle_artifact_removal_widgets
        )
        self.use_crf = ui.CheckBox(
            "Use CRF post-processing",
            func=self._toggle_display_crf,
            parent=self,
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

    def _toggle_crf_choice(self):
        if self.model_choice.currentText() == "WNet":
            self.use_crf.setVisible(True)
        else:
            self.use_crf.setVisible(False)

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
        self.use_window_choice.setToolTip(
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
        artifact_tooltip = "If enabled, will remove labels of objects larger than the chosen size in instance segmentation"
        self.artifact_removal_size.setToolTip(
            artifact_tooltip + "\nDefault is 500 pixels"
        )
        self.attempt_artifact_removal_box.setToolTip(artifact_tooltip)
        self.artifact_container.setToolTip(artifact_tooltip)
        ##################
        ##################

    def check_ready(self):
        """Checks if the paths to the files are properly set."""
        if self.layer_choice.isChecked():
            if self.image_layer_loader.layer_data() is not None:
                return True
        elif (
            self.folder_choice.isChecked()
            and self.image_filewidget.check_ready()
        ):
            return True
        return False

    def _restrict_window_size_for_model(self):
        """Sets the window size to a value that is compatible with the chosen model."""
        self.wnet_enabled = False
        if self.model_choice.currentText() == "WNet":
            self.wnet_enabled = True
            self.window_size_choice.setCurrentIndex(self._default_window_size)
            self.use_window_choice.setChecked(self.wnet_enabled)
        self.window_size_choice.setDisabled(
            self.wnet_enabled and not self.custom_weights_choice.isChecked()
        )
        self.use_window_choice.setDisabled(
            self.wnet_enabled and not self.custom_weights_choice.isChecked()
        )

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
        """Shows the choices for viewing results depending on whether :py:attr:`self.view_checkbox` is checked."""
        ui.toggle_visibility(self.view_checkbox, self.view_results_container)

    def _toggle_display_thresh(self):
        """Shows the choices for thresholding results depending on whether :py:attr:`self.thresholding_checkbox` is checked."""
        ui.toggle_visibility(
            self.thresholding_checkbox, self.thresholding_slider.container
        )

    def _toggle_display_artifact_size_thresh(self):
        """Shows the choices for thresholding results depending on whether :py:attr:`self.attempt_artifact_removal_box` is checked."""
        ui.toggle_visibility(
            self.attempt_artifact_removal_box,
            self.artifact_removal_size,
        )
        ui.toggle_visibility(
            self.attempt_artifact_removal_box,
            self.remove_artifacts_label,
        )

    def _toggle_display_crf(self):
        """Shows the choices for CRF post-processing depending on whether :py:attr:`self.use_crf` is checked."""
        ui.toggle_visibility(self.use_crf, self.crf_widgets)

    def _toggle_display_instance(self):
        """Shows or hides the options for instance segmentation based on current user selection."""
        ui.toggle_visibility(self.use_instance_choice, self.instance_widgets)

    def _toggle_artifact_removal_widgets(self):
        """Shows or hides the options for instance segmentation based on current user selection."""
        ui.toggle_visibility(self.use_instance_choice, self.artifact_container)
        ui.toggle_visibility(
            self.use_instance_choice, self.attempt_artifact_removal_box
        )

    def _toggle_display_window_size(self):
        """Show or hide window size choice depending on status of self.window_infer_box."""
        ui.toggle_visibility(self.use_window_choice, self.window_infer_params)

    def _load_weights_path(self):
        """Show file dialog to set :py:attr:`model_path`."""
        # logger.debug(self._default_weights_folder)

        file = ui.open_file_dialog(
            self,
            [self._default_weights_folder],
            file_extension="Weights file (*.pth *.pt *.onnx)",
        )
        self._update_weights_path(file)

    def _build(self):
        """Puts all widgets in a layout and adds them to the napari Viewer."""
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
                self.use_window_choice,
                self.window_infer_params,
                self.keep_data_on_cpu_box,
                self.device_choice.label,
                self.device_choice,
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
            "Additional options", parent=self
        )

        self.thresholding_slider.container.setVisible(False)

        ui.add_widgets(
            self.artifact_container.layout,
            [
                self.attempt_artifact_removal_box,
                self.remove_artifacts_label,
                self.artifact_removal_size,
            ],
        )
        self.attempt_artifact_removal_box.setVisible(False)
        self.remove_artifacts_label.setVisible(False)
        self.artifact_removal_size.setVisible(False)

        ui.add_widgets(
            post_proc_layout,
            [
                self.anisotropy_wdgt,  # anisotropy
                self.thresholding_checkbox,
                self.thresholding_slider.container,  # thresholding
                self.use_crf,
                self.crf_widgets,
                self.use_instance_choice,
                self.instance_widgets,
                self.artifact_container,
                self.save_stats_to_csv_box,
                # self.instance_param_container,  # instance segmentation
            ],
        )
        self._toggle_crf_choice()
        self.model_choice.currentIndexChanged.connect(self._toggle_crf_choice)
        ModelFramework._show_io_element(
            self.save_stats_to_csv_box, self.use_instance_choice
        )

        self.anisotropy_wdgt.container.setVisible(False)
        self.thresholding_slider.container.setVisible(False)
        self.instance_widgets.setVisible(False)
        self.crf_widgets.setVisible(False)
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

    def _update_progress_bar(self, image_id: int, total: int):
        pbar_value = image_id // total
        if pbar_value == 0:
            pbar_value = 1

        self.progress.setValue(100 * pbar_value)

    def _display_results(self, result: InferenceResult):
        viewer = self._viewer
        if self.worker_config.post_process_config.zoom.enabled:
            zoom = self.worker_config.post_process_config.zoom.zoom_values
        else:
            zoom = [1, 1, 1]
        image_id = result.image_id
        model_name = self.model_choice.currentText()

        # viewer.dims.ndisplay = 3 # let user choose
        viewer.scale_bar.visible = True

        if self.config.show_original and result.original is not None:
            viewer.add_image(
                result.original,
                colormap="inferno",
                name=f"original_{image_id}",
                scale=zoom,
                opacity=0.7,
            )

        out_colormap = "turbo"
        # if self.worker_config.post_process_config.thresholding.enabled:
        # out_colormap = "twilight"

        viewer.add_image(
            result.semantic_segmentation,
            colormap=out_colormap,
            name=f"pred_{image_id}_{model_name}",
            opacity=0.8,
        )

        if (
            len(result.semantic_segmentation.shape) == 4
        ):  # seek channel that is most likely to be foreground
            fractions_per_channel = utils.channels_fraction_above_threshold(
                result.semantic_segmentation, 0.5
            )
            index_channel_sorted = np.argsort(fractions_per_channel)
            for channel in index_channel_sorted:
                if result.semantic_segmentation[channel].sum() > 0:
                    index_channel_least_labelled = channel
                    break
            viewer.dims.set_point(
                0, index_channel_least_labelled
            )  # TODO(cyril: check if this is always the right axis

        if result.crf_results is not None and not isinstance(
            result.crf_results, Exception
        ):
            logger.debug(f"CRF results shape : {result.crf_results.shape}")
            viewer.add_image(
                result.crf_results,
                name=f"CRF_results_image_{image_id}",
                colormap="viridis",
            )
        if (
            result.instance_labels is not None
            and not isinstance(result.instance_labels, Exception)
            and self.worker_config.post_process_config.instance.enabled
        ):
            method_name = (
                self.worker_config.post_process_config.instance.method.name
            )

            if len(result.instance_labels.shape) >= 4:
                channels_by_labels = np.argsort(
                    result.instance_labels.sum(axis=(1, 2, 3))
                )
                min_objs_channel = channels_by_labels[0]
                # if least labeled is empty, use next least labeled channel
                for i in range(1, len(channels_by_labels)):
                    if (
                        np.unique(
                            result.instance_labels[
                                channels_by_labels[i]
                            ].flatten()
                        ).size
                        > 1
                    ):
                        min_objs_channel = channels_by_labels[i]
                        break

                number_cells = (
                    np.unique(
                        result.instance_labels[min_objs_channel].flatten()
                    ).size
                    - 1
                )
            else:
                number_cells = (
                    np.unique(result.instance_labels.flatten()).size - 1
                )  # remove background with -1

            name = f"({number_cells} objects)_{method_name}_instance_labels_{image_id}"

            viewer.add_labels(result.instance_labels, name=name)

            if result.stats is not None and isinstance(
                result.stats, list
            ):  # list for several channels
                logger.debug(f"len stats : {len(result.stats)}")

                for i, stats in enumerate(result.stats):
                    # stats = result.stats

                    if self.worker_config.compute_stats and stats is not None:
                        try:
                            stats_dict = stats.get_dict()
                            stats_df = pd.DataFrame(stats_dict)

                            self.log.print_and_log(
                                f"Number of instances in channel {i} : {stats.number_objects[0]}"
                            )

                            csv_name = f"/{model_name}_{method_name}_seg_results_{image_id}_channel_{i}_{utils.get_date_time()}.csv"

                            stats_df.to_csv(
                                self.worker_config.results_path + csv_name,
                                index=False,
                            )
                        except ValueError as e:
                            logger.warning(f"Error saving stats to csv : {e}")
                            logger.debug(
                                f"Length of stats array : {[len(s) for s in stats.get_dict().values()]}"
                            )
                            # logger.debug(f"Stats dict : {stats.get_dict()}")

    def _setup_worker(self):
        if self.folder_choice.isChecked():
            self.worker_config.images_filepaths = self.images_filepaths
            self.worker = InferenceWorker(worker_config=self.worker_config)

        elif self.layer_choice.isChecked():
            self.worker_config.layer = self.image_layer_loader.layer()
            self.worker = InferenceWorker(worker_config=self.worker_config)

        else:
            raise ValueError("Please select to load a layer or folder")

        self.worker.set_download_log(self.log)
        self.worker.started.connect(self.on_start)

        self.worker.log_signal.connect(self.log.print_and_log)
        self.worker.log_w_replace_signal.connect(self.log.replace_last_line)
        self.worker.warn_signal.connect(self.log.warn)
        self.worker.error_signal.connect(self.log.error)

        self.worker.yielded.connect(partial(self.on_yield))
        self.worker.errored.connect(partial(self.on_error))
        self.worker.finished.connect(self.on_finish)

        if self.get_device(show=False) == "cuda":
            self.worker.finished.connect(self.empty_cuda_cache)
        return self.worker

    def start(self):
        """Start the inference process, enables :py:attr:`~self.worker` and does the following.

        * Checks if the output and input folders are correctly set

        * Loads the weights from the chosen model

        * Creates a dict with all image paths (see :py:func:`~create_inference_dict`)

        * Loads the images, pads them so their size is a power of two in every dim (see :py:func:`utils.get_padding_dim`)

        * Performs sliding window inference (from MONAI) on every image, with or without ROI of chosen size

        * Saves all outputs in the selected results folder

        * If the option has been selected, display the results in napari, up to the maximum number selected

        * Runs instance segmentation, thresholding, and stats computing if requested
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
            self._set_worker_config()
            if self.worker_config is None:
                raise RuntimeError("Worker config was not set correctly")
            self._setup_worker()
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

    def _create_worker_from_config(
        self, worker_config: config.InferenceWorkerConfig
    ):
        if isinstance(worker_config, config.InfererConfig):
            raise TypeError("Please provide a valid worker config object")
        return InferenceWorker(worker_config=worker_config)

    def _set_self_config(self):
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

    def _set_worker_config(self) -> config.InferenceWorkerConfig:
        self.model_info = config.ModelInfo(
            name=self.model_choice.currentText(),
            model_input_size=self.model_input_size.value(),
        )

        self.weights_config.use_custom = self.custom_weights_choice.isChecked()

        save_path = self.results_filewidget.text_field.text()
        if not self._check_results_path(save_path):
            msg = f"ERROR: please set valid results path. Current path is {save_path}"
            self.log.print_and_log(msg)
            logger.warning(msg)
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

        self.instance_config = config.InstanceSegConfig(
            enabled=self.use_instance_choice.isChecked(),
            method=self.instance_widgets.methods[
                self.instance_widgets.method_choice.currentText()
            ],
        )
        self.instance_config.method.record_parameters()  # keep parameters set when Start is clicked

        self.post_process_config = config.PostProcessConfig(
            zoom=zoom_config,
            thresholding=thresholding_config,
            instance=self.instance_config,
            artifact_removal=self.attempt_artifact_removal_box.isChecked(),
            artifact_removal_size=self.artifact_removal_size.value(),
        )

        if self.use_window_choice.isChecked():
            size = int(self.window_size_choice.currentText())
            window_config = config.SlidingWindowConfig(
                window_size=size,
                window_overlap=self.window_overlap_slider.slider_value,
            )
        else:
            window_config = config.SlidingWindowConfig()

        self.worker_config = config.InferenceWorkerConfig(
            device=self.check_device_choice(),
            model_info=self.model_info,
            weights_config=self.weights_config,
            results_path=self.results_path,
            filetype=".tif",
            keep_on_cpu=self.keep_data_on_cpu_box.isChecked(),
            compute_stats=self.save_stats_to_csv_box.isChecked(),
            post_process_config=self.post_process_config,
            sliding_window_config=window_config,
            use_crf=self.use_crf.isChecked(),
            crf_config=self.crf_widgets.make_config(),
        )
        return self.worker_config

    def on_start(self):
        """Catches start signal from worker to call :py:func:`~display_status_report`."""
        self.display_status_report()
        self._set_self_config()
        self.log.print_and_log(f"Worker started at {utils.get_time()}")
        self.log.print_and_log(f"Saving results to : {self.results_path}")
        self.log.print_and_log("Worker is running...")

    def on_error(self, error):
        """Catches errors and tries to clean up."""
        self.log.print_and_log("!" * 20)
        self.log.print_and_log("Worker errored...")
        self.log.error(error)
        self.worker.quit()
        self.on_finish()

    def on_finish(self):
        """Catches finished signal from worker, resets workspace for next run."""
        self.log.print_and_log(f"\nWorker finished at {utils.get_time()}")
        self.log.print_and_log("*" * 20)
        self.btn_start.setText("Start")
        self.btn_close.setVisible(True)

        self.worker = None
        self.worker_config = None
        self.empty_cuda_cache()
        return True  # signal clean exit

    def on_yield(self, result: InferenceResult):
        """Displays the inference results in napari.

        Works as long as data["image_id"] is lower than nbr_to_show, and updates the status report docked widget (namely the progress bar).

        Args:
            result (InferenceResult): results from the worker
        """
        if isinstance(result, Exception):
            self.on_error(result)
            # raise result
        if result is None:
            self.on_error("Worker yielded None")
        # viewer, progress, show_res, show_res_number, zoon, show_original

        # check that viewer checkbox is on and that max number of displays has not been reached.
        # widget.log.print_and_log(result)
        try:
            image_id = result.image_id
            if self.worker_config.images_filepaths is not None:
                total = len(self.worker_config.images_filepaths)
            else:
                total = 1
            self._update_progress_bar(image_id, total)

            if (
                self.config.show_results
                and image_id <= self.config.show_results_count
            ):
                self._display_results(result)
        except Exception as e:
            self.on_error(e)
