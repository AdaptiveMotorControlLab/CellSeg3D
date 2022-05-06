import os
import warnings
import numpy as np

import napari
# Qt
from qtpy.QtWidgets import QLabel
from qtpy.QtWidgets import QSizePolicy
from qtpy.QtWidgets import QWidget

# local
from napari_cellseg3d import interface as ui
from napari_cellseg3d import model_instance_seg as inst_seg
from napari_cellseg3d import utils
from napari_cellseg3d.model_framework import ModelFramework
from napari_cellseg3d.model_workers import InferenceWorker


class Inferer(ModelFramework):
    """A plugin to run already trained models in evaluation mode to preform inference and output a label on all
    given volumes."""

    def __init__(self, viewer: "napari.viewer.Viewer"):
        """
        Creates an Inference loader plugin with the following widgets :

        * Data :
            * A file extension choice for the images to load from selected folders

            * Two buttons to choose the images folder to run segmentation and save results in, respectively

        * Post-processing :
            * A box to select if data is anisotropic, if checked, asks for resolution in micron for each axis

            * A box to choose whether to threshold, if checked asks for a threshold between 0 and 1

        * Display options :
            * A dropdown menu to select which model should be used for inference

            * A checkbox to choose whether to display results in napari afterwards. Will ask for how many results to display, capped at 10

        * A button to launch the inference process

        * A button to close the widget

        TODO:

        * Verify if way of loading model is  OK

        * Padding OK ?

        * Save toggle ?

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

        ############################
        ############################
        ############################
        ############################
        # TEST TODO REMOVE
        import glob

        directory = "C:/Users/Cyril/Desktop/test/test"

        # self.data_path = directory

        self.images_filepaths = sorted(
            glob.glob(os.path.join(directory, "*.tif"))
        )
        self.results_path = "C:/Users/Cyril/Desktop/test"

        #######################
        #######################
        #######################

        ###########################
        # interface

        self.view_checkbox = ui.make_checkbox(
            "View results in napari", self.toggle_display_number
        )

        self.display_number_choice = ui.make_n_spinboxes(1, 1, 10, 1)
        self.lbl_display_number = QLabel("How many ? (max. 10)", self)

        self.aniso_checkbox = ui.make_checkbox(
            "Anisotropic data", self.toggle_display_aniso
        )

        self.aniso_box_widgets = ui.make_n_spinboxes(
            n=3, min=1.0, default=1.5, step=0.5, double=True
        )
        self.aniso_box_lbl = [
            QLabel("Resolution in " + axis + " (microns) :") for axis in "xyz"
        ]

        self.aniso_box_widgets[-1].setValue(5.0)  # TODO change default

        for w in self.aniso_box_widgets:
            w.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.aniso_resolutions = []

        self.thresholding_checkbox = ui.make_checkbox(
            "Perform thresholding", self.toggle_display_thresh
        )

        self.thresholding_count = ui.make_n_spinboxes(
            n=1, max=1, default=0.7, step=0.05, double=True
        )

        ##################
        ##################
        # instance segmentation widgets
        self.instance_method_dict = (
            {  # TODO move conversion to callable in workers
                "Connected components": inst_seg.binary_connected,
                "Watershed": inst_seg.binary_watershed,
            }
        )
        self.instance_choice = ui.make_checkbox(
            "Run instance segmentation", func=self.toggle_display_instance
        )

        self.instance_method_choice = ui.make_combobox(
            sorted(self.instance_method_dict.keys())
        )

        self.instance_prob_thresh = ui.make_n_spinboxes(
            n=1, max=0.99, default=0.7, step=0.05, double=True
        )
        self.instance_prob_thresh_lbl = QLabel("Probability threshold :")
        self.instance_prob_t_container = ui.combine_blocks(
            second=self.instance_prob_thresh,
            first=self.instance_prob_thresh_lbl,
            horizontal=False,
        )

        self.instance_small_object_thresh = ui.make_n_spinboxes(
            n=1, max=100, default=10, step=5
        )
        self.instance_small_object_thresh_lbl = QLabel(
            "Small object removal threshold :"
        )
        self.instance_small_object_t_container = ui.combine_blocks(
            second=self.instance_small_object_thresh,
            first=self.instance_small_object_thresh_lbl,
            horizontal=False,
        )
        ##################
        ##################
        self.show_original_checkbox = ui.make_checkbox("Show originals")

        self.btn_start = ui.make_button("Start inference", self.start)
        self.btn_close = self.make_close_button()

        # hide unused widgets from parent class
        self.btn_label_files.setVisible(False)
        self.lbl_label_files.setVisible(False)
        self.btn_model_path.setVisible(False)
        self.lbl_model_path.setVisible(False)

        self.build()

    def check_ready(self):
        """Checks if the paths to the files are properly set"""
        if (
            self.images_filepaths != [""]
            and self.images_filepaths != []
            and self.results_path != ""
        ):
            return True
        else:
            warnings.formatwarning = utils.format_Warning
            warnings.warn("Image and label paths are not correctly set")
            return False

    def toggle_display_number(self):
        """Shows the choices for viewing results depending on whether :py:attr:`self.view_checkbox` is checked"""
        if self.view_checkbox.isChecked():
            self.display_number_choice.setVisible(True)
            self.lbl_display_number.setVisible(True)
            self.show_original_checkbox.setVisible(True)
        else:
            self.display_number_choice.setVisible(False)
            self.lbl_display_number.setVisible(False)
            self.show_original_checkbox.setVisible(False)

    def toggle_display_aniso(self):
        """Shows the choices for correcting anisotropy when viewing results depending on whether :py:attr:`self.aniso_checkbox` is checked"""
        if self.aniso_checkbox.isChecked():
            for w, lbl in zip(self.aniso_box_widgets, self.aniso_box_lbl):
                w.setVisible(True)
                lbl.setVisible(True)
        else:
            for w, lbl in zip(self.aniso_box_widgets, self.aniso_box_lbl):
                w.setVisible(False)
                lbl.setVisible(False)

    def toggle_display_thresh(self):
        """Shows the choices for thresholding results depending on whether :py:attr:`self.thresholding_checkbox` is checked"""
        if self.thresholding_checkbox.isChecked():
            self.thresholding_count.setVisible(True)
        else:
            self.thresholding_count.setVisible(False)

    def toggle_display_instance(self):
        """Shows or hides the options for instance segmentation based on current user selection"""
        if self.instance_choice.isChecked():
            self.instance_method_choice.setVisible(True)
            self.instance_prob_t_container.setVisible(True)
            self.instance_small_object_t_container.setVisible(True)
        else:
            self.instance_method_choice.setVisible(False)
            self.instance_prob_t_container.setVisible(False)
            self.instance_small_object_t_container.setVisible(False)

    def build(self):
        """Puts all widgets in a layout and adds them to the napari Viewer"""

        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.MinimumExpanding)
        ######
        ############
        ##################
        tab, tab_layout = ui.make_container_widget(
            0, 0, 1, 1
        )  # tab that will contain all widgets

        L, T, R, B = 7, 20, 7, 11  # margins for group boxes
        #################################
        #################################
        io_group, io_layout = ui.make_group("Data", L, T, R, B)

        io_layout.addWidget(
            ui.combine_blocks(self.filetype_choice, self.lbl_filetype),
            alignment=ui.LEFT_AL,
        )  # file extension
        io_layout.addWidget(
            ui.combine_blocks(self.btn_image_files, self.lbl_image_files),
            alignment=ui.LEFT_AL,
        )  # in folder
        io_layout.addWidget(
            ui.combine_blocks(self.btn_result_path, self.lbl_result_path),
            alignment=ui.LEFT_AL,
        )  # out folder

        io_group.setLayout(io_layout)
        tab_layout.addWidget(io_group, alignment=ui.LEFT_AL)
        #################################
        #################################
        ui.add_blank(self, tab_layout)
        #################################
        #################################
        # model group

        model_group_w, model_group_l = ui.make_group(
            "Model choice",
            L,
            T,
            R,
            B,
        )  # model choice

        model_group_l.addWidget(self.model_choice, alignment=ui.LEFT_AL)
        model_group_l.addWidget(
            self.custom_weights_choice, alignment=ui.LEFT_AL
        )
        model_group_l.addWidget(
            self.weights_path_container, alignment=ui.LEFT_AL
        )
        self.weights_path_container.setVisible(False)
        self.lbl_model_choice.setVisible(False)  # TODO remove

        model_group_w.setLayout(model_group_l)
        tab_layout.addWidget(model_group_w, alignment=ui.LEFT_AL)

        #################################
        #################################
        ui.add_blank(self, tab_layout)
        #################################
        #################################
        # post proc group
        post_proc_group, post_proc_layout = ui.make_group(
            "Post-processing", L, T, R, B
        )

        post_proc_layout.addWidget(self.aniso_checkbox, alignment=ui.LEFT_AL)

        [
            post_proc_layout.addWidget(widget, alignment=ui.LEFT_AL)
            for wdgts in zip(self.aniso_box_lbl, self.aniso_box_widgets)
            for widget in wdgts
        ]
        for w in self.aniso_box_widgets:
            w.setVisible(False)
        for w in self.aniso_box_lbl:
            w.setVisible(False)
        # anisotropy
        ui.add_blank(post_proc_group, post_proc_layout)

        post_proc_layout.addWidget(
            self.thresholding_checkbox, alignment=ui.LEFT_AL
        )
        post_proc_layout.addWidget(
            self.thresholding_count, alignment=ui.CENTER_AL
        )
        self.thresholding_count.setVisible(False)  # thresholding

        ui.add_blank(post_proc_group, post_proc_layout)

        # instance segmentation
        post_proc_layout.addWidget(self.instance_choice, alignment=ui.LEFT_AL)
        post_proc_layout.addWidget(
            self.instance_method_choice, alignment=ui.LEFT_AL
        )
        post_proc_layout.addWidget(
            self.instance_prob_t_container, alignment=ui.LEFT_AL
        )
        post_proc_layout.addWidget(
            self.instance_small_object_t_container, alignment=ui.LEFT_AL
        )

        self.instance_method_choice.setVisible(False)
        self.instance_prob_t_container.setVisible(False)
        self.instance_small_object_t_container.setVisible(False)

        post_proc_group.setLayout(post_proc_layout)
        tab_layout.addWidget(post_proc_group, alignment=ui.LEFT_AL)
        ###################################
        ###################################
        ui.add_blank(self, tab_layout)
        ###################################
        ###################################
        display_opt_group, display_opt_layout = ui.make_group(
            "Display options", L, T, R, B
        )

        display_opt_layout.addWidget(
            self.view_checkbox,  # ui.combine_blocks(self.view_checkbox, self.lbl_view),
            alignment=ui.LEFT_AL,
        )  # view_after bool
        display_opt_layout.addWidget(
            self.lbl_display_number, alignment=ui.LEFT_AL
        )
        display_opt_layout.addWidget(
            self.display_number_choice,
            alignment=ui.LEFT_AL,
        )  # number of results to display
        display_opt_layout.addWidget(
            self.show_original_checkbox,
            alignment=ui.LEFT_AL,
        )  # show original bool
        self.show_original_checkbox.toggle()

        self.display_number_choice.setVisible(False)
        self.show_original_checkbox.setVisible(False)
        self.lbl_display_number.setVisible(False)

        # TODO : add custom model handling ? using exec() to read user provided model class
        # self.lbl_label.setText("model.pth directory :")

        display_opt_group.setLayout(display_opt_layout)
        tab_layout.addWidget(display_opt_group)
        ###################################
        ui.add_blank(self, tab_layout)
        ###################################
        ###################################
        tab_layout.addWidget(self.btn_start, alignment=ui.LEFT_AL)
        tab_layout.addWidget(self.btn_close, alignment=ui.LEFT_AL)
        ##################
        ############
        ######
        # end of tabs, combine into scrollable
        ui.make_scrollable(
            containing_widget=tab,
            contained_layout=tab_layout,
            min_wh=[200, 100],
        )
        self.addTab(tab, "Inference")

    def start(self):
        """Start the inference process, enables :py:attr:`~self.worker` and does the following:

        * Checks if the output and input folders are correctly set

        * Loads the weights from the chosen model

        * Creates a dict with all image paths (see :py:func:`~create_inference_dict`)

        * Loads the images, pads them so their size is a power of two in every dim (see :py:func:`utils.get_padding_dim`)

        * Performs sliding window inference (from MONAI) on every image

        * Saves all outputs in the selected results folder

        * If the option has been selected, display the results in napari, up to the maximum number selected
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
                self.btn_start.setText("Running... Click to stop")
        else:
            self.log.print_and_log("Starting...")
            self.log.print_and_log("*" * 20)

            device = self.get_device()

            model_key = self.model_choice.currentText()
            model_dict = {  # gather model info
                "name": model_key,
                "class": self.get_model(model_key),
            }

            if self.custom_weights_choice.isChecked():
                weights_dict = {"custom": True, "path": self.weights_path}
            else:
                weights_dict = {
                    "custom": False,
                    "path": self.get_model(model_key).get_weights_file(),
                }

            if self.aniso_checkbox.isChecked():
                self.aniso_resolutions = [
                    w.value() for w in self.aniso_box_widgets
                ]
                self.zoom = utils.anisotropy_zoom_factor(
                    self.aniso_resolutions
                )
            else:
                self.zoom = [1, 1, 1]

            self.transforms = {  # TODO figure out a better way ?
                "thresh": [
                    self.thresholding_checkbox.isChecked(),
                    self.thresholding_count.value(),
                ],
                "zoom": [
                    self.aniso_checkbox.isChecked(),
                    self.zoom,
                ],
            }

            method_call = self.instance_method_dict[
                self.instance_method_choice.currentText()
            ]
            prob_thresh = self.instance_prob_thresh.value()
            t_small = self.instance_small_object_thresh.value()
            if self.instance_method_choice.currentText() == "Watershed":
                method = lambda image: method_call(
                    image, thres_seeding=prob_thresh, thres_small=t_small
                )
            elif (
                self.instance_method_choice.currentText()
                == "Connected components"
            ):
                method = lambda image: method_call(image, prob_thresh, t_small)
            else:
                print(
                    "Error, method for instance segmentation is incorrectly set"
                )

            self.instance_params = {
                "do_instance": self.instance_choice.isChecked(),
                "method": method,
                "name": self.instance_method_choice.currentText(),
            }

            self.show_res_nbr = self.display_number_choice.value()

            self.worker = InferenceWorker(
                device=device,
                model_dict=model_dict,
                weights_dict=weights_dict,
                images_filepaths=self.images_filepaths,
                results_path=self.results_path,
                filetype=self.filetype_choice.currentText(),
                transforms=self.transforms,
                instance=self.instance_params,
            )

            yield_connect_show_res = lambda data: self.on_yield(
                data,
                widget=self,
            )

            self.worker.started.connect(self.on_start)
            self.worker.log_signal.connect(self.log.print_and_log)
            self.worker.yielded.connect(yield_connect_show_res)
            self.worker.errored.connect(yield_connect_show_res)  # TODO fix
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

        self.show_res = self.view_checkbox.isChecked()
        self.show_original = self.show_original_checkbox.isChecked()
        self.log.print_and_log(f"Worker started at {utils.get_time()}")
        self.log.print_and_log(f"Saving results to : {self.results_path}")
        self.log.print_and_log("Worker is running...")

        if self.transforms["zoom"][0]:
            self.log.print_and_log(
                f"\nAnisotropy parameters are : {self.aniso_resolutions} microns in x,y,z"
            )

        if self.instance_params["do_instance"]:
            self.log.print_and_log(f"\nInstance segmentation is enabled")
            self.log.print_and_log(
                f"Instance segmentation method is : {self.instance_params['name']}"
            )
            self.log.print_and_log(
                f"Probability threshold is {self.instance_prob_thresh.value()}"
            )
            self.log.print_and_log(
                f"Small object threshold is {self.instance_small_object_thresh.value()}"
            )

    def on_error(self):
        """Catches errors and tries to clean up. TODO : upgrade"""
        self.log.print_and_log("Worker errored...")
        self.log.print_and_log("Trying to clean up...")
        self.btn_start.setText("Start")
        self.btn_close.setVisible(True)

        self.worker = None
        self.empty_cuda_cache()

    def on_finish(self):
        """Catches finished signal from worker, resets workspace for next run."""
        self.log.print_and_log(f"\nWorker finished at {utils.get_time()}")
        self.log.print_and_log("*" * 20)
        self.btn_start.setText("Start")
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
        # check that viewer checkbox is on and that max number of displays has not been reached.
        image_id = data["image_id"]
        model_name = data["model_name"]
        total = len(widget.images_filepaths)

        viewer = widget._viewer

        widget.progress.setValue(100 * (image_id) // total)

        if widget.show_res and image_id <= widget.show_res_nbr:

            zoom = widget.zoom

            print(data["original"].shape)
            print(data["result"].shape)

            viewer.dims.ndisplay = 3
            viewer.scale_bar.visible = True

            if widget.show_original:
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

                widget.log.print_and_log(f"\nNUMBER OF CELLS : {np.amax(data['instance_labels'])}\n")

                name = f"instance_labels_{image_id}"
                instance_layer = viewer.add_labels(
                    data["instance_labels"], name=name
                )
