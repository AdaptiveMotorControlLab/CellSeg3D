import os
from datetime import datetime
from pathlib import Path

import napari
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
    LabelFilter,
    SpatialPadd,
)
# local
from napari_cellseg_annotator import utils
from napari_cellseg_annotator.model_framework import ModelFramework
from napari_cellseg_annotator.models import model_SegResNet as SegResNet
from napari_cellseg_annotator.models import model_VNet as VNet
# Qt
from qtpy.QtWidgets import (
    QVBoxLayout,
    QPushButton,
    QSizePolicy,
    QLabel,
    QCheckBox,
    QComboBox,
)
from tifffile import imwrite

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

        * Add option to choose number of images to display in napari (1,3,all)

        * Save toggle ?

        * Prevent launch if not all args are set to avoid undefined behaviour

        Args:
            viewer (napari.viewer.Viewer): napari viewer to display the widget in
        """
        super().__init__(viewer)

        self._viewer = viewer

        self.models_dict = {"VNet": VNet, "SegResNet": SegResNet}
        """dict: dictionary of available models, with string for widget display as key
        
        Currently implemented : SegResNet, VNet"""

        self.view_checkbox = QCheckBox()
        self.view_checkbox.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.lbl_view = QLabel("View in napari after prediction ?", self)

        self.model_choice = QComboBox()
        self.model_choice.addItems(sorted(self.models_dict.keys()))
        self.lbl_model_choice = QLabel("Model name", self)

        self.btn_start = QPushButton("Start inference")
        self.btn_start.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn_start.clicked.connect(self.start)

        #hide unused widgets from parent class
        self.btn_label_files.setVisible(False)
        self.lbl_label_files.setVisible(False)
        self.btn_model_path.setVisible(False)
        self.lbl_model_path.setVisible(False)

        #####################################################################
        # TODO remove once done
        self.test_button = True
        if self.test_button:
            self.btntest = QPushButton("test", self)
            self.btntest.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            self.btntest.clicked.connect(self.run_test)
        #####################################################################

        self.get_device()
        self.build()

    def get_model(self, key):
        """Getter for module associated to currently selected model"""
        return self.models_dict[key]

    def create_inference_dict(self):
        """Create a dict with all image paths in :py:attr:`self.images_filepaths`

        Returns:
            dict: list of image paths from loaded folder"""
        data_dicts = [
            {"image": image_name} for image_name in self.images_filepaths
        ]
        return data_dicts

    def build(self):
        """Build buttons in a layout and add them to the napari Viewer"""
        vbox = QVBoxLayout()

        vbox.addWidget(
            utils.combine_blocks(self.filetype_choice, self.lbl_filetype)
        )  # file extension
        vbox.addWidget(
            utils.combine_blocks(self.btn_image_files, self.lbl_image_files)
        )  # in folder
        vbox.addWidget(
            utils.combine_blocks(self.btn_result_path, self.lbl_result_path)
        )  # out folder

        vbox.addWidget(
            utils.combine_blocks(self.model_choice, self.lbl_model_choice)
        )  # model choice
        vbox.addWidget(
            utils.combine_blocks(self.view_checkbox, self.lbl_view)
        )  # view_after bool

        # TODO : add custom model handling ? using exec() to read user provided model class
        # self.lbl_label.setText("model.pth directory :")

        vbox.addWidget(self.btn_start)
        vbox.addWidget(self.btn_close)

        ##################################################################
        # TODO remove once done ?

        if self.test_button:
            vbox.addWidget(self.btntest)
        ##################################################################

        self.setLayout(vbox)

        ########################
        # TODO : remove once done

    def run_test(self):

        self.images_filepaths = [
            "C:/Users/Cyril/Desktop/Proj_bachelor/data/visual_tif/volumes/images.tif"
        ]
        filenames = self.images_filepaths
        path = os.path.dirname(filenames[0])
        self.lbl_image_files.setText(path)
        self.update_default()

        self.view_checkbox.toggle()

        self.results_path = "C:/Users/Cyril/Desktop/test"
        self.lbl_result_path.setText(self.results_path)
        self.update_default()
        self.start()

    # self.close()

    ########################

    def start(self):
        """Start the inference process and does the following:

        * Loads the weights from the chosen model

        * Creates a dict with all image paths (see :py:func:`create_inference_dict`)

        * Loads the images, pads them so their size is a power of two in every dim (see :py:func:`get_padding_dim`)

        * Performs sliding window inference (from MONAI) on every image

        * Saves all outputs in the selected results folder

        * If the option has been selected, display the results in napari

        """
        device = self.device

        model_key = self.model_choice.currentText()

        model = self.get_model(model_key).get_net()

        model.to(device)

        images_dict = self.create_inference_dict()

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
            AsDiscrete(threshold=0.1),
            LabelFilter(applied_labels=[0]),
        )

        inference_ds = Dataset(data=images_dict, transform=load_transforms)
        inference_loader = DataLoader(
            inference_ds, batch_size=1, num_workers=4
        )

        weights = self.get_model(model_key).get_weights_file()
        # print(f"wh dir : {WEIGHTS_DIR}")
        # print(weights)
        model.load_state_dict(torch.load(os.path.join(WEIGHTS_DIR, weights), map_location=device))

        model.eval()
        with torch.no_grad():
            for i, inf_data in enumerate(inference_loader):

                inputs = inf_data["image"]
                inputs = inputs.to(device)
                outputs = sliding_window_inference(
                    inputs, roi_size= None,sw_batch_size= 1, predictor= lambda inputs: model(inputs)[0], device = device
                )

                out = outputs.detach().cpu()
                out = post_process_transforms(out)
                out = np.array(out).astype(np.float32)
                print(f"Saving to : {self.results_path}")

                time = "{:%Y_%m_%d_%H_%M_%S}".format(datetime.now())
                print(time)
                filename = (
                    self.results_path
                    + "/"
                    + self.model_choice.currentText()
                    + f"_{time}_"
                    + f"pred{i}"
                    + self.filetype_choice.currentText()
                )

                # print(filename)
                imwrite(filename, out)

                print(f"File n°{i} saved as :")
                print(filename)

                if self.view_checkbox.isChecked():

                    viewer = self._viewer

                    in_data = np.array(inf_data["image"]).astype(np.float32)

                    original_layer = viewer.add_image(
                        in_data,
                        colormap="inferno",
                        name=f"original_{i}",
                        scale=[1, 1, 1],
                        opacity=0.7,
                    )

                    out_layer = viewer.add_image(
                        out[0],
                        colormap="gist_earth",
                        name=f"pred_{i}",
                        opacity=0.8,
                    )

        return
