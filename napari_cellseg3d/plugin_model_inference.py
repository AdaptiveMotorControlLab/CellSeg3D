import warnings

import napari
import numpy as np
import pandas as pd

# Qt
from qtpy.QtWidgets import QSizePolicy

# local
from napari_cellseg3d import interface as ui
from napari_cellseg3d import utils
from napari_cellseg3d.model_framework import ModelFramework
from napari_cellseg3d.model_workers import InferenceWorker


# TODO for layer inference : button behaviour/visibility, error if no layer selected, test all funcs


class Inferer(ModelFramework):
    """A plugin to run already trained models in evaluation mode to preform inference and output a label on all
    given volumes."""

    def __init__(self, viewer: "napari.viewer.Viewer"):
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
        super().__init__(viewer)

        self._viewer = viewer
        """Viewer to display the widget in"""

        self.worker = None
        """Worker for inference, should be an InferenceWorker instance from :doc:model_workers.py"""

        self.transforms = None

        self.show_res = False
        self.show_res_nbr = 1
        self.show_original = True
        self.zoom = [1, 1, 1]

        self.instance_params = None
        self.stats_to_csv = False

        self.keep_on_cpu = False
        self.use_window_inference = False
        self.window_inference_size = None
        self.window_overlap = 0.25

        ###########################
        # interface

        (
            self.view_results_container,
            self.view_results_layout,
        ) = ui.make_container(T=7, B=0, parent=self)

        self.view_checkbox = ui.make_checkbox(
            "View results in napari", self.toggle_display_number
        )

        self.display_number_choice = ui.IntIncrementCounter(min=1, default=5)
        self.lbl_display_number = ui.make_label("How many ? (max. 10)", self)

        self.show_original_checkbox = ui.make_checkbox("Show originals")

        ######################
        ######################
        # TODO : better way to handle SegResNet size reqs ?
        self.model_input_size = ui.IntIncrementCounter(
            min=1, max=1024, default=128
        )
        self.model_choice.currentIndexChanged.connect(
            self.toggle_display_model_input_size
        )
        self.model_choice.setCurrentIndex(0)

        self.anisotropy_wdgt = ui.AnisotropyWidgets(
            self,
            default_x=1.5,
            default_y=1.5,
            default_z=5,  # TODO change default
        )

        self.aniso_resolutions = [1, 1, 1]

        # ui.add_blank(self.aniso_container, aniso_layout)

        ######################
        ######################
        self.thresholding_checkbox = ui.make_checkbox(
            "Perform thresholding", self.toggle_display_thresh
        )

        self.thresholding_count = ui.DoubleIncrementCounter(
            max=1, default=0.7, step=0.05
        )

        self.thresholding_container, self.thresh_layout = ui.make_container(
            T=7, parent=self
        )

        self.window_infer_box = ui.CheckBox(title="Use window inference")
        self.window_infer_box.clicked.connect(self.toggle_display_window_size)

        sizes_window = ["8", "16", "32", "64", "128", "256", "512"]
        # (
        #     self.window_size_choice,
        #     self.lbl_window_size_choice,
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
        self.lbl_window_size_choice = self.window_size_choice.label

        self.window_overlap_counter = ui.DoubleIncrementCounter(
            min=0,
            max=1,
            default=0.25,
            step=0.05,
            parent=self,
            label="Overlap %",
        )

        self.keep_data_on_cpu_box = ui.CheckBox(title="Keep data on CPU")

        window_size_widgets = ui.combine_blocks(
            self.window_size_choice,
            self.lbl_window_size_choice,
            horizontal=False,
        )
        # self.window_infer_params = ui.combine_blocks(
        #     self.window_overlap,
        #     self.window_infer_params,
        #     horizontal=False,
        # )

        self.window_infer_params = ui.combine_blocks(
            window_size_widgets,
            self.window_overlap_counter.get_with_label(horizontal=False),
            horizontal=False,
        )

        ##################
        ##################
        # instance segmentation widgets
        self.instance_box = ui.make_checkbox(
            "Run instance segmentation", func=self.toggle_display_instance
        )

        self.instance_method_choice = ui.DropdownMenu(
            ["Connected components", "Watershed"]
        )

        self.instance_prob_thresh = ui.DoubleIncrementCounter(
            max=0.99, default=0.7, step=0.05
        )
        self.instance_prob_thresh_lbl = ui.make_label(
            "Probability threshold :", self
        )
        self.instance_prob_t_container = ui.combine_blocks(
            right_or_below=self.instance_prob_thresh,
            left_or_above=self.instance_prob_thresh_lbl,
            horizontal=False,
        )

        self.instance_small_object_thresh = ui.IntIncrementCounter(
            max=100, default=10, step=5
        )
        self.instance_small_object_thresh_lbl = ui.make_label(
            "Small object removal threshold :", self
        )
        self.instance_small_object_t_container = ui.combine_blocks(
            right_or_below=self.instance_small_object_thresh,
            left_or_above=self.instance_small_object_thresh_lbl,
            horizontal=False,
        )
        self.save_stats_to_csv_box = ui.make_checkbox(
            "Save stats to csv", parent=self
        )

        (
            self.instance_param_container,
            self.instance_layout,
        ) = ui.make_container(T=7, B=0, parent=self)

        ##################
        ##################

        self.btn_start = ui.Button("Start on folder", self.start)
        self.btn_start_layer = ui.Button(
            "Start on selected layer",
            lambda: self.start(on_layer=True),
        )
        self.btn_close = self.make_close_button()

        # hide unused widgets from parent class
        self.label_filewidget.setVisible(False)
        self.model_filewidget.setVisible(False)

        ##################
        ##################
        # tooltips
        self.view_checkbox.setToolTip("Show results in the napari viewer")
        self.display_number_choice.setToolTip(
            "Choose how many results to display once the work is done.\n"
            "Maximum is 10 for clarity"
        )
        self.show_original_checkbox.setToolTip(
            "Displays the image used for inference in the viewer"
        )
        self.model_input_size.setToolTip(
            "Image size on which the model has been trained (default : 128)"
        )

        thresh_desc = (
            "Thresholding : all values in the image below the chosen probability"
            " threshold will be set to 0, and all others to 1."
        )

        self.thresholding_checkbox.setToolTip(thresh_desc)
        self.thresholding_count.setToolTip(thresh_desc)
        self.window_infer_box.setToolTip(
            "Sliding window inference runs the model on parts of the image"
            "\nrather than the whole image, to reduce memory requirements."
            "\nUse this if you have large images."
        )
        self.window_size_choice.setToolTip(
            "Size of the window to run inference with (in pixels)"
        )

        self.window_overlap_counter.setToolTip(
            "Percentage of overlap between windows to use when using sliding window"
        )

        # self.window_overlap.setToolTip(
        #     "Amount of overlap between sliding windows"
        # )

        self.keep_data_on_cpu_box.setToolTip(
            "If enabled, data will be kept on the RAM rather than the VRAM.\nCan avoid out of memory issues with CUDA"
        )
        self.instance_box.setToolTip(
            "Instance segmentation will convert instance (0/1) labels to labels"
            " that attempt to assign an unique ID to each cell."
        )
        self.instance_method_choice.setToolTip(
            "Choose which method to use for instance segmentation"
            "\nConnected components : all separated objects will be assigned an unique ID. "
            "Robust but will not work correctly with adjacent/touching objects\n"
            "Watershed : assigns objects ID based on the probability gradient surrounding an object. "
            "Requires the model to surround objects in a gradient;"
            " can possibly correctly separate unique but touching/adjacent objects."
        )
        self.instance_prob_thresh.setToolTip(
            "All objects below this probability will be ignored (set to 0)"
        )
        self.instance_small_object_thresh.setToolTip(
            "Will remove all objects smaller (in volume) than the specified number of pixels"
        )
        self.save_stats_to_csv_box.setToolTip(
            "Will save several statistics for each object to a csv in the results folder. Stats include : "
            "volume, centroid coordinates, sphericity"
        )
        ##################
        ##################

        self.build()

    def check_ready(self):
        """Checks if the paths to the files are properly set"""
        if (
            self.images_filepaths != [""]
            and self.images_filepaths != []
            and self.results_path != ""
        ) or (
            self.results_path != ""
            and self._viewer.layers.selection.active is not None
        ):
            return True
        else:
            warnings.formatwarning = utils.format_Warning
            warnings.warn("Image and label paths are not correctly set")
            return False

    def toggle_display_model_input_size(self):
        if (
            self.model_choice.currentText() == "SegResNet"
            or self.model_choice.currentText() == "SwinUNetR"
        ):
            self.model_input_size.setVisible(True)
        else:
            self.model_input_size.setVisible(False)

    def toggle_display_number(self):
        """Shows the choices for viewing results depending on whether :py:attr:`self.view_checkbox` is checked"""
        ui.toggle_visibility(self.view_checkbox, self.view_results_container)

    def toggle_display_thresh(self):
        """Shows the choices for thresholding results depending on whether :py:attr:`self.thresholding_checkbox` is checked"""
        ui.toggle_visibility(
            self.thresholding_checkbox, self.thresholding_container
        )

    def toggle_display_instance(self):
        """Shows or hides the options for instance segmentation based on current user selection"""
        ui.toggle_visibility(self.instance_box, self.instance_param_container)

    def toggle_display_window_size(self):
        """Show or hide window size choice depending on status of self.window_infer_box"""
        ui.toggle_visibility(self.window_infer_box, self.window_infer_params)

    def build(self):
        """Puts all widgets in a layout and adds them to the napari Viewer"""

        # ui.add_blank(self.view_results_container, view_results_layout)
        ui.add_widgets(
            self.view_results_layout,
            [
                self.view_checkbox,
                self.lbl_display_number,
                self.display_number_choice,
                self.show_original_checkbox,
            ],
            alignment=None,
        )

        self.view_results_container.setLayout(self.view_results_layout)

        self.anisotropy_wdgt.build()

        self.thresh_layout.addWidget(
            self.thresholding_count, alignment=ui.LEFT_AL
        )
        # ui.add_blank(self.thresholding_container, thresh_layout)
        self.thresholding_container.setLayout(
            self.thresh_layout
        )  # thresholding
        self.thresholding_container.setVisible(False)

        ui.add_widgets(
            self.instance_layout,
            [
                self.instance_method_choice,
                self.instance_prob_t_container,
                self.instance_small_object_t_container,
                self.save_stats_to_csv_box,
            ],
        )

        self.instance_param_container.setLayout(self.instance_layout)

        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.MinimumExpanding)
        ######
        ############
        ##################
        tab, tab_layout = ui.make_container(
            B=1, parent=self
        )  # tab that will contain all widgets

        L, T, R, B = 7, 20, 7, 11  # margins for group boxes
        #################################
        #################################
        io_group, io_layout = ui.make_group("Data", L, T, R, B, parent=self)

        ui.add_widgets(
            io_layout,
            [
                ui.combine_blocks(
                    self.filetype_choice, self.lbl_filetype
                ),  # file extension
                ui.combine_blocks(
                    self.btn_image_files, self.lbl_image_files
                ),  # in folder
                ui.combine_blocks(
                    self.btn_result_path, self.lbl_result_path
                ),  # out folder
            ],
        )
        self.image_filewidget.set_required(False)
        self.image_filewidget.update_field_color("black")

        io_group.setLayout(io_layout)
        tab_layout.addWidget(io_group)
        #################################
        #################################
        ui.add_blank(tab, tab_layout)
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
                self.weights_path_container,
                self.model_input_size,
            ],
        )
        self.weights_path_container.setVisible(False)
        self.lbl_model_choice.setVisible(False)  # TODO remove (?)

        model_group_w.setLayout(model_group_l)
        tab_layout.addWidget(model_group_w)

        #################################
        #################################
        ui.add_blank(tab, tab_layout)
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

        tab_layout.addWidget(inference_param_group_w)

        #################################
        #################################
        ui.add_blank(tab, tab_layout)
        #################################
        #################################
        # post proc group
        post_proc_group, post_proc_layout = ui.make_group(
            "Post-processing", parent=self
        )

        ui.add_widgets(
            post_proc_layout,
            [
                self.anisotropy_wdgt,  # anisotropy
                self.thresholding_checkbox,
                self.thresholding_container,  # thresholding
                self.instance_box,
                self.instance_param_container,  # instance segmentation
            ],
        )

        self.anisotropy_wdgt.container.setVisible(False)
        self.thresholding_container.setVisible(False)
        self.instance_param_container.setVisible(False)

        post_proc_group.setLayout(post_proc_layout)
        tab_layout.addWidget(post_proc_group, alignment=ui.LEFT_AL)
        ###################################
        ###################################
        ui.add_blank(tab, tab_layout)
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
        self.toggle_display_number()

        # TODO : add custom model handling ?
        # self.lbl_label.setText("model.pth directory :")

        display_opt_group.setLayout(display_opt_layout)
        tab_layout.addWidget(display_opt_group)
        ###################################
        ui.add_blank(self, tab_layout)
        ###################################
        ###################################
        ui.add_widgets(
            tab_layout,
            [
                self.btn_start,
                self.btn_start_layer,
                self.btn_close,
            ],
        )
        ##################
        ############
        ######
        # end of tabs, combine into scrollable
        ui.ScrollArea.make_scrollable(
            parent=tab,
            contained_layout=tab_layout,
            min_wh=[200, 100],
        )
        self.addTab(tab, "Inference")

        self.setMinimumSize(180, 100)
        # self.setBaseSize(210, 400)

    def start(self, on_layer=False):
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
            err = "Aborting, please choose correct paths"
            self.log.print_and_log(err)
            raise ValueError(err)

        if self.worker is not None:
            if self.worker.is_running:
                pass
            else:
                self.worker.start()
                self.btn_start_layer.setVisible(False)
                self.btn_start.setText("Running... Click to stop")
        else:
            self.log.print_and_log("Starting...")
            self.log.print_and_log("*" * 20)

            device = self.get_device()

            model_key = self.model_choice.currentText()
            model_dict = {  # gather model info
                "name": model_key,
                "class": self.get_model(model_key),
                "model_input_size": self.model_input_size.value(),
            }

            if self.custom_weights_choice.isChecked():
                weights_dict = {"custom": True, "path": self.weights_path}
            else:
                weights_dict = {
                    "custom": False,
                }

            if self.anisotropy_wdgt.is_enabled():
                self.aniso_resolutions = (
                    self.anisotropy_wdgt.get_anisotropy_resolution_xyz(
                        as_factors=False
                    )
                )
                self.zoom = (
                    self.anisotropy_wdgt.get_anisotropy_resolution_xyz()
                )
            else:
                self.zoom = [1, 1, 1]

            self.transforms = {  # TODO figure out a better way ?
                "thresh": [
                    self.thresholding_checkbox.isChecked(),
                    self.thresholding_count.value(),
                ],
                "zoom": [
                    self.anisotropy_wdgt.checkbox.isChecked(),
                    self.zoom,
                ],
            }

            self.instance_params = {
                "do_instance": self.instance_box.isChecked(),
                "method": self.instance_method_choice.currentText(),
                "threshold": self.instance_prob_thresh.value(),
                "size_small": self.instance_small_object_thresh.value(),
            }
            self.stats_to_csv = self.save_stats_to_csv_box.isChecked()
            # print(f"METHOD : {self.instance_method_choice.currentText()}")

            self.show_res_nbr = self.display_number_choice.value()

            self.keep_on_cpu = self.keep_data_on_cpu_box.isChecked()
            self.use_window_inference = self.window_infer_box.isChecked()
            self.window_inference_size = int(
                self.window_size_choice.currentText()
            )
            self.window_overlap = self.window_overlap_counter.value()

            if not on_layer:
                self.worker = InferenceWorker(
                    device=device,
                    model_dict=model_dict,
                    weights_dict=weights_dict,
                    images_filepaths=self.images_filepaths,
                    results_path=self.results_path,
                    filetype=self.filetype_choice.currentText(),
                    transforms=self.transforms,
                    instance=self.instance_params,
                    use_window=self.use_window_inference,
                    window_infer_size=self.window_inference_size,
                    window_overlap=self.window_overlap,
                    keep_on_cpu=self.keep_on_cpu,
                    stats_csv=self.stats_to_csv,
                )
            else:
                layer = self._viewer.layers.selection.active
                self.worker = InferenceWorker(
                    device=device,
                    model_dict=model_dict,
                    weights_dict=weights_dict,
                    results_path=self.results_path,
                    filetype=self.filetype_choice.currentText(),
                    transforms=self.transforms,
                    instance=self.instance_params,
                    use_window=self.use_window_inference,
                    window_infer_size=self.window_inference_size,
                    keep_on_cpu=self.keep_on_cpu,
                    window_overlap=self.window_overlap,
                    stats_csv=self.stats_to_csv,
                    layer=layer,
                )

            self.worker.set_download_log(self.log)

            yield_connect_show_res = lambda data: self.on_yield(
                data,
                widget=self,
            )

            self.worker.started.connect(self.on_start)
            self.worker.log_signal.connect(self.log.print_and_log)
            self.worker.warn_signal.connect(self.log.warn)
            self.worker.yielded.connect(yield_connect_show_res)
            self.worker.errored.connect(
                yield_connect_show_res
            )  # TODO fix showing errors from thread
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
            self.btn_start_layer.setVisible(False)

    def on_start(self):
        """Catches start signal from worker to call :py:func:`~display_status_report`"""
        self.display_status_report()

        self.show_res = self.view_checkbox.isChecked()
        self.show_original = self.show_original_checkbox.isChecked()
        self.log.print_and_log(f"Worker started at {utils.get_time()}")
        self.log.print_and_log(f"Saving results to : {self.results_path}")
        self.log.print_and_log("Worker is running...")

    def on_error(self):
        """Catches errors and tries to clean up. TODO : upgrade"""
        self.log.print_and_log("Worker errored...")
        self.log.print_and_log("Trying to clean up...")
        self.btn_start.setText("Start on folder")
        self.btn_close.setVisible(True)

        self.worker = None
        self.empty_cuda_cache()

    def on_finish(self):
        """Catches finished signal from worker, resets workspace for next run."""
        self.log.print_and_log(f"\nWorker finished at {utils.get_time()}")
        self.log.print_and_log("*" * 20)
        self.btn_start.setText("Start on folder")
        self.btn_start_layer.setVisible(True)
        self.btn_close.setVisible(True)

        self.worker = None
        self.empty_cuda_cache()

    @staticmethod
    def on_yield(data, widget):
        """
        Displays the inference results in napari as long as data["image_id"] is lower than nbr_to_show,
        and updates the status report docked widget (namely the progress bar)

        Args:
            data (dict): dict yielded by :py:func:`~inference()`, contains : "image_id" : index of the returned image, "original" : original volume used for inference, "result" : inference result
            widget (QWidget): widget for accessing attributes
        """
        # viewer, progress, show_res, show_res_number, zoon, show_original

        # check that viewer checkbox is on and that max number of displays has not been reached.
        image_id = data["image_id"]
        model_name = data["model_name"]
        total = len(widget.images_filepaths)

        viewer = widget._viewer

        pbar_value = image_id // total
        if image_id == 0:
            pbar_value = 1

        widget.progress.setValue(100 * pbar_value)

        if widget.show_res and image_id <= widget.show_res_nbr:

            zoom = widget.zoom

            viewer.dims.ndisplay = 3
            viewer.scale_bar.visible = True

            if widget.show_original and data["original"] is not None:
                original_layer = viewer.add_image(
                    data["original"],
                    colormap="inferno",
                    name=f"original_{image_id}",
                    scale=zoom,
                    opacity=0.7,
                )

            out_colormap = "twilight"
            if widget.transforms["thresh"][0]:
                out_colormap = "turbo"

            out_layer = viewer.add_image(
                data["result"],
                colormap=out_colormap,
                name=f"pred_{image_id}_{model_name}",
                opacity=0.8,
            )

            if data["instance_labels"] is not None:

                labels = data["instance_labels"]
                method = widget.instance_params["method"]
                number_cells = np.amax(labels)

                name = f"({number_cells} objects)_{method}_instance_labels_{image_id}"

                instance_layer = viewer.add_labels(labels, name=name)

                data_dict = data["object stats"]

                if widget.stats_to_csv and data_dict is not None:

                    numeric_data = pd.DataFrame(data_dict)

                    csv_name = f"/{method}_seg_results_{image_id}_{utils.get_date_time()}.csv"
                    numeric_data.to_csv(
                        widget.results_path + csv_name, index=False
                    )

                    # widget.log.print_and_log(
                    #     f"\nNUMBER OF CELLS : {number_cells}\n"
                    # )
