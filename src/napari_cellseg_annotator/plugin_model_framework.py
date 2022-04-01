import glob
import os
import numpy as np
import napari
from napari_cellseg_annotator.plugin_base import BasePlugin

from monai.utils import first
import monai.transforms as mtf

from monai.networks.nets import SegResNetVAE
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.data import (
    CacheDataset,
    DataLoader,
    Dataset,
    decollate_batch,
    pad_list_data_collate,
)
import torch


class ModelFramework(BasePlugin):
    def __init__(self, viewer: "napari.viewer.Viewer"):

        super().__init__(viewer)

        self.model_type = None

        self.data_filepaths = []
        self.label_filepaths = []
        self.results_path = ""

        self.train_image_transforms = mtf.Compose(  # ?
        [
            mtf.LoadImaged(keys=["image", "label"]),
            # AddChanneld(keys=["image", "label"]), #already done
            mtf.EnsureChannelFirstd(keys=["image", "label"]),
            mtf.RandSpatialCropd(keys=["image", "label"], roi_size=[64, 64, 64]),
            mtf.SpatialPadd(keys=["image", "label"], spatial_size=[128, 128, 128]),
            mtf.RandShiftIntensityd(keys=["image"], offsets=0.2),
            mtf.EnsureTyped(keys=["image", "label"]),
        ]
    )

        #######################################################
        # interface

        #######################################################

    def load_dataset_paths(self, directory, filetype):
        filenames = []

        for filename in glob.glob(os.path.join(directory, "*" + filetype)):
            filenames.append(filename)

        images_paths = sorted(glob.glob(os.path.join(directory, "*.tif")))

        return images_paths

    def create_dataset_dict(self, images_paths, labels_paths):

        data_dicts = [
            {"image": image_name, "label": label_name}
            for image_name, label_name in zip(images_paths, labels_paths)
        ]

        return data_dicts

    def transform(self):
        return

    def train(self):
        return
