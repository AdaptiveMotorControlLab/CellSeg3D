import os
import warnings
from pathlib import Path

import napari
import numpy as np
import torch
from monai.data import DataLoader
from monai.data import Dataset
# MONAI
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete
from monai.transforms import Compose
from monai.transforms import EnsureChannelFirstd
from monai.transforms import EnsureType
from monai.transforms import EnsureTyped
from monai.transforms import LoadImaged
from monai.transforms import SpatialPadd
from monai.transforms import Zoomd
from napari.qt.threading import thread_worker
from qtpy.QtCore import Qt
# Qt
from qtpy.QtWidgets import QCheckBox
from qtpy.QtWidgets import QLabel
from qtpy.QtWidgets import QLayout
from qtpy.QtWidgets import QPushButton
from qtpy.QtWidgets import QSizePolicy
from qtpy.QtWidgets import QSpinBox
from qtpy.QtWidgets import QVBoxLayout
from qtpy.QtWidgets import QWidget

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

        self.view_checkbox = QCheckBox()
        self.view_checkbox.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.view_checkbox.stateChanged.connect(self.toggle_display_number)
        self.lbl_view = QLabel("View results in napari ?", self)

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

    @staticmethod
    def create_inference_dict(images_filepaths):
        """Create a dict with all image paths in :py:attr:`self.images_filepaths`

        Returns:
            dict: list of image paths from loaded folder"""
        data_dicts = [{"image": image_name} for image_name in images_filepaths]
        return data_dicts

    def check_ready(self):
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
            utils.combine_blocks(self.filetype_choice, self.lbl_filetype),
            alignment=Qt.AlignmentFlag.AlignLeft,
        )  # file extension
        tab_layout.addWidget(
            utils.combine_blocks(self.btn_image_files, self.lbl_image_files),
            alignment=Qt.AlignmentFlag.AlignLeft,
        )  # in folder
        tab_layout.addWidget(
            utils.combine_blocks(self.btn_result_path, self.lbl_result_path),
            alignment=Qt.AlignmentFlag.AlignLeft,
        )  # out folder

        tab_layout.addWidget(
            utils.combine_blocks(self.model_choice, self.lbl_model_choice),
            alignment=Qt.AlignmentFlag.AlignLeft,
        )  # model choice
        tab_layout.addWidget(
            utils.combine_blocks(self.view_checkbox, self.lbl_view),
            alignment=Qt.AlignmentFlag.AlignLeft,
        )  # view_after bool
        tab_layout.addWidget(
            utils.combine_blocks(
                self.display_number_choice, self.lbl_display_number
            ),
            alignment=Qt.AlignmentFlag.AlignLeft,
        )
        self.display_number_choice.setVisible(False)
        self.lbl_display_number.setVisible(False)

        # TODO : add custom model handling ? using exec() to read user provided model class
        # self.lbl_label.setText("model.pth directory :")
        utils.add_blank(self, tab_layout)

        tab_layout.addWidget(
            self.btn_start, alignment=Qt.AlignmentFlag.AlignLeft
        )
        tab_layout.addWidget(
            self.btn_close, alignment=Qt.AlignmentFlag.AlignLeft
        )

        tab.setLayout(tab_layout)
        self.addTab(tab, "Inference")

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

            device = self.get_device()

            model_key = self.model_choice.currentText()
            model_dict = {
                "name": model_key,
                "class": self.get_model(model_key).get_net(),
            }

            weights = self.get_model(model_key).get_weights_file()

            self.worker = self.inference(
                device,
                model_dict,
                weights,
                self.images_filepaths,
                self.results_path,
                self.filetype_choice.currentText(),
            )

            self.worker.started.connect(
                lambda: print("\nWorker is running...")
            )

            yield_connect = lambda data: self.show_results(
                data,
                viewer=self._viewer,
                nbr_to_show=self.display_number_choice.value(),
                show=self.view_checkbox.isChecked(),
            )
            self.worker.yielded.connect(yield_connect)

            self.worker.finished.connect(lambda: print("Worker finished"))
            self.worker.finished.connect(
                lambda: self.btn_start.setText("Start")
            )
            self.worker.finished.connect(
                lambda: self.btn_close.setVisible(True)
            )

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

    @staticmethod
    def show_results(data, viewer, nbr_to_show=0, show=False):
        # check that viewer checkbox is on and that max number of displays has not been reached.
        image_id = data["image_id"]
        if show and image_id <= nbr_to_show:

            viewer.dims.ndisplay = 3
            viewer.scale_bar.visible = True
            original_layer = viewer.add_image(
                data["original"],
                colormap="inferno",
                name=f"original_{image_id}",
                scale=[1, 1, 1],
                opacity=0.7,
            )

            out_layer = viewer.add_image(
                data["result"],
                colormap="twilight_shifted",
                name=f"pred_{image_id}",
                opacity=0.8,
            )

    @staticmethod
    @thread_worker
    def inference(
        device, model_dict, weights, images_filepaths, results_path, filetype
    ):

        model = model_dict["class"]
        model.to(device)

        images_dict = Inferer.create_inference_dict(images_filepaths)

        # TODO : better solution than loading first image always ?
        data = LoadImaged(keys=["image"])(images_dict[0])
        # print(data)
        check = data["image"].shape
        # print(check)
        # TODO remove
        z_aniso = 5 / 1.5
        pad = utils.get_padding_dim(check, anisotropy_factor=[1, 1, z_aniso])
        # print(pad)

        # TODO : add toggle
        anisotropic_transform = Zoomd(
            keys=["image"],
            zoom=(1, 1, 1 / z_aniso),
            keep_size=False,
            padding_mode="empty",
        )

        load_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                # AddChanneld(keys=["image"]), #already done
                EnsureChannelFirstd(keys=["image"]),
                # Orientationd(keys=["image"], axcodes="PLI"),
                anisotropic_transform,
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

                print(f"Saving to : {results_path}")

                image_id = i + 1
                time = utils.get_date_time()
                # print(time)

                original_filename = os.path.basename(
                    images_filepaths[i]
                ).split(".")[0]

                # File output save name : original-name_model_date+time_number.filetype
                filename = (
                    results_path
                    + "/"
                    + original_filename
                    + "_"
                    + model_dict["name"]
                    + f"_{time}_"
                    + f"pred_{image_id}"
                    + filetype
                )

                # print(filename)
                # imwrite(filename, out)

                print(f"File n°{image_id} saved as :")
                print(filename)

                original = np.array(inf_data["image"]).astype(np.float32)

                yield {"image_id": i + 1, "original": original, "result": out}
