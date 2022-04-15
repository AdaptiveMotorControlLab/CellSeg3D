import os
import warnings
from pathlib import Path

import napari
from napari.qt.threading import thread_worker
import numpy as np
import torch
from monai.data import (
    DataLoader,
    Dataset,
)

# MONAI
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    EnsureTyped,
    EnsureType,
    SpatialPadd,
)

# Qt
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QSizePolicy,
    QLabel,
    QCheckBox,
    QSpinBox,
    QLayout,
)
from tifffile import imwrite

# local
from napari_cellseg_annotator import utils
from napari_cellseg_annotator.model_framework import ModelFramework

WEIGHTS_DIR = os.path.dirname(os.path.realpath(__file__)) + str(
    Path("/models/saved_weights")
)


class Inferer(ModelFramework):
    """A plugin to run already trained models in evaluation mode to preform inference and output a label on all
    given volumes."""

    def __init__(self, viewer: "napari.viewer.Viewer"):
        """
        Creates an Inference loader plugin with the following widgets :

        * A filetype choice for the images to load

        * Two buttons to choose the images folder to run segmentation and save results in, respectively

        * A dropdown menu to select which model should be used for inference

        * A checkbox to choose whether to display results in napari afterwards

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

        self.worker = None
        """Worker for inference"""
        self.inf_data = []

        self.image_id = []

        self.out = []

        self.view_checkbox = QCheckBox()
        self.view_checkbox.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.view_checkbox.stateChanged.connect(self.toggle_display_number)
        self.lbl_view = QLabel(
            "View results in napari after prediction ?", self
        )

        self.display_number_choice = QSpinBox()
        self.display_number_choice.setRange(1, 10)
        self.display_number_choice.setSizePolicy(
            QSizePolicy.Fixed, QSizePolicy.Fixed
        )
        self.lbl_display_number = QLabel("How many ? (max. 10)", self)

        self.btn_start = QPushButton("Start inference")
        self.btn_start.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn_start.clicked.connect(self.start)

        # hide unused widgets from parent class
        self.btn_label_files.setVisible(False)
        self.lbl_label_files.setVisible(False)
        self.btn_model_path.setVisible(False)
        self.lbl_model_path.setVisible(False)

        self.build()

    def create_inference_dict(self):
        """Create a dict with all image paths in :py:attr:`self.images_filepaths`

        Returns:
            dict: list of image paths from loaded folder"""
        data_dicts = [
            {"image": image_name} for image_name in self.images_filepaths
        ]
        return data_dicts

    def check_ready(self):
        if self.images_filepaths != [""] and self.results_path != "":
            return True
        else:
            warnings.formatwarning = utils.format_Warning
            warnings.warn("Image and label paths are not correctly set")
            return False

    def toggle_display_number(self):
        if self.view_checkbox.isChecked():
            self.display_number_choice.setVisible(True)
            self.lbl_display_number.setVisible(True)
        else:
            self.display_number_choice.setVisible(False)
            self.lbl_display_number.setVisible(False)

    def build(self):
        """Build buttons in a layout and add them to the napari Viewer"""

        tab = QWidget()
        tab_layout = QVBoxLayout()
        tab_layout.setSizeConstraint(QLayout.SetFixedSize)

        tab_layout.addWidget(
            utils.combine_blocks(self.filetype_choice, self.lbl_filetype)
        )  # file extension
        tab_layout.addWidget(
            utils.combine_blocks(self.btn_image_files, self.lbl_image_files)
        )  # in folder
        tab_layout.addWidget(
            utils.combine_blocks(self.btn_result_path, self.lbl_result_path)
        )  # out folder

        tab_layout.addWidget(
            utils.combine_blocks(self.model_choice, self.lbl_model_choice)
        )  # model choice
        tab_layout.addWidget(
            utils.combine_blocks(self.view_checkbox, self.lbl_view)
        )  # view_after bool
        tab_layout.addWidget(
            utils.combine_blocks(
                self.display_number_choice, self.lbl_display_number
            )
        )
        self.display_number_choice.setVisible(False)
        self.lbl_display_number.setVisible(False)

        # TODO : add custom model handling ? using exec() to read user provided model class
        # self.lbl_label.setText("model.pth directory :")
        tab_layout.addWidget(QLabel("", self))
        tab_layout.addWidget(self.btn_start)
        tab_layout.addWidget(self.btn_close)

        tab.setLayout(tab_layout)
        self.addTab(tab, "Inference")

    # @staticmethod #, show, inf_data, image_id, out
    def show_results(self):
        viewer = self._viewer
        # check that viewer checkbox is on and that max number of displays has not been reached.
        for inf_data, out, image_id in zip(self.inf_data, self.out, self.image_id) :

            original = np.array(inf_data["image"]).astype(np.float32)
            print("why")
            original_layer = viewer.add_image(
                original,
                colormap="inferno",
                name=f"original_{image_id}",
                scale=[1, 1, 1],
                opacity=0.7,
            )

            out_layer = viewer.add_image(
                out,
                colormap="twilight_shifted",
                name=f"pred_{image_id}",
                opacity=0.8,
            )
        self.inf_data = self.out = self.image_id = []

    def start(self):
        """Start the inference process and does the following:

        * Checks if the output and input folders are correctly set

        * Loads the weights from the chosen model

        * Creates a dict with all image paths (see :py:func:`create_inference_dict`)

        * Loads the images, pads them so their size is a power of two in every dim (see :py:func:`get_padding_dim`)

        * Performs sliding window inference (from MONAI) on every image

        * Saves all outputs in the selected results folder

        * If the option has been selected, display the results in napari, up to the maximum number selected

        TODO:

        * Turn prediction into a function ? (for threading maybe)

        * Use os.makedirs(dir, exist_ok = True) ?

        * Multithreading ?

        """

        if not self.check_ready():
            raise ValueError("Aborting, please choose correct paths")

        if self.worker is not None:
            if self.worker.is_running:
                pass
            else:
                self.worker.start()
                self.btn_start.setText("Running... Click to stop")
        else:

            self.worker = self.inference()
            self.worker.started.connect(lambda: print("\nWorker is running..."))
            self.worker.finished.connect(lambda: print("Worker finished"))
            self.worker.finished.connect(
                lambda: self.btn_start.setText("Start")
            )
            self.worker.finished.connect(
                lambda: self.btn_close.setVisible(True)
            )
            self.worker.finished.connect(self.reset_worker)
            self.worker.finished.connect(self.show_results)


            if self.get_device(show=False) == "cuda":
                self.worker.finished.connect(self.empty_cuda_cache)
            self.btn_close.setVisible(False)

        if self.worker.is_running:
            print(
                "Stop request, waiting for next inference & saving to occur..."
            )
            self.btn_start.setText("Stopping...")
            self.worker.quit()
        else:
            self.worker.start()
            self.btn_start.setText("Running...  Click to stop")

    def reset_worker(self):
        self.worker= None

    @thread_worker
    def inference(self):

        device = self.get_device()

        model_key = self.model_choice.currentText()

        model = self.get_model(model_key).get_net()

        model.to(device)

        images_dict = self.create_inference_dict()

        # TODO : better solution than loading first image always ?
        data = LoadImaged(keys=["image"])(images_dict[0])
        # print(data)
        check = data["image"].shape
        # print(check)
        pad = self.get_padding_dim(check)
        # print(pad)

        load_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                # AddChanneld(keys=["image"]), #already done
                EnsureChannelFirstd(keys=["image"]),
                SpatialPadd(keys=["image"], spatial_size=pad),
                EnsureTyped(keys=["image"]),
            ]
        )
        post_process_transforms = Compose(
            EnsureType(),
            AsDiscrete(threshold=0.8),
            # LabelFilter(applied_labels=[0]),
        )

        inference_ds = Dataset(data=images_dict, transform=load_transforms)
        inference_loader = DataLoader(
            inference_ds, batch_size=1, num_workers=4
        )

        weights = self.get_model(model_key).get_weights_file()
        # print(f"wh dir : {WEIGHTS_DIR}")
        # print(weights)
        model.load_state_dict(
            torch.load(os.path.join(WEIGHTS_DIR, weights), map_location=device)
        )

        # use multithreading ?
        model.eval()
        with torch.no_grad():
            for i, inf_data in enumerate(inference_loader):

                inputs = inf_data["image"]
                inputs = inputs.to(device)
                outputs = sliding_window_inference(
                    inputs,
                    roi_size=None,
                    sw_batch_size=3,
                    predictor=lambda inputs: model(inputs)[0],
                    device=device,
                )

                out = outputs.detach().cpu()
                out = post_process_transforms(out)
                out = np.array(out).astype(np.float32)
                print(f"Saving to : {self.results_path}")

                image_id = i + 1
                time = utils.get_date_time()
                # print(time)

                original_filename = os.path.basename(
                    self.images_filepaths[i]
                ).split(".")[0]

                # File output save name : original-name_model_date+time_number.filetype
                filename = (
                    self.results_path
                    + "/"
                    + original_filename
                    + "_"
                    + self.model_choice.currentText()
                    + f"_{time}_"
                    + f"pred_{image_id}"
                    + self.filetype_choice.currentText()
                )

                # print(filename)
                # imwrite(filename, out)

                print(f"File n°{image_id} saved as :")
                print(filename)



                if self.view_checkbox.isChecked() and i < self.display_number_choice.value():
                    self.inf_data.append(inf_data)
                    self.image_id.append(image_id)
                    print(out)
                    self.out.append(out[0])

                yield


