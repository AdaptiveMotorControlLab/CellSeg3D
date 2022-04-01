import os
from datetime import datetime
import napari
import torch
import numpy as np

from pathlib import Path
from tifffile import imwrite

from qtpy.QtWidgets import (
    QVBoxLayout,
    QPushButton,
    QSizePolicy,
    QLabel,
    QCheckBox,
    QComboBox,
)

from monai.utils import first
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
from monai.data import (
    DataLoader,
    Dataset,
)

from napari_cellseg_annotator import utils
from napari_cellseg_annotator.model_framework import ModelFramework
from napari_cellseg_annotator.models import model_SegResNet as SegResNet


WEIGHTS_DIR = os.path.dirname(os.path.realpath(__file__)) + str(
    Path("/models/saved_weights")
)


class Inferer(ModelFramework):
    """A plugin to run already trained models in evaluation mode to preform inference and output a volume label."""

    def __init__(self, viewer: "napari.viewer.Viewer"):

        super().__init__(viewer)

        self._viewer = viewer

        self.models_dict = {"VNet": " ", "SegResNet": SegResNet}
        self.current_model = None

        self.view_checkbox = QCheckBox()
        self.view_checkbox.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.lbl_view = QLabel("View in napari after prediction ?", self)

        self.model_choice = QComboBox()
        self.model_choice.addItems(sorted(self.models_dict.keys()))
        self.lbl_model_choice = QLabel("Model name", self)

        self.btn_start = QPushButton("Start inference")
        self.btn_start.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn_start.clicked.connect(self.start)

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
        return self.models_dict[key]

    def create_inference_dict(self):
        data_dicts = [
            {"image": image_name} for image_name in self.images_filepaths
        ]
        return data_dicts

    def build(self):

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

        device = self.device

        post_process_transforms = Compose(
            EnsureType(),
            AsDiscrete(threshold=0.1),
            LabelFilter(applied_labels=[0]),
        )

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

        inference_ds = Dataset(data=images_dict, transform=load_transforms)
        inference_loader = DataLoader(
            inference_ds, batch_size=1, num_workers=4
        )

        weights = self.get_model(model_key).get_weights_file()
        # print(f"wh dir : {WEIGHTS_DIR}")
        # print(weights)
        model.load_state_dict(torch.load(os.path.join(WEIGHTS_DIR, weights)))

        model.eval()
        with torch.no_grad():
            for i, inf_data in enumerate(inference_loader):

                inputs = inf_data["image"]
                inputs = inputs.to(device)
                outputs = sliding_window_inference(
                    inputs, None, 1, lambda inputs: model(inputs)[0]
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

                print(filename)
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
                        opacity=0.6,
                    )
        if self.view_checkbox.isChecked():
            napari.run()

        return
