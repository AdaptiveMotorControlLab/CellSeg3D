import glob
import os

import napari
from napari_cellseg_annotator.plugin_base import BasePlugin


class ModelFramework(BasePlugin):
    def __init__(self, viewer: "napari.viewer.Viewer"):

        super().__init__(viewer)

        self.model_type = None

        self.data_filepaths = []
        self.label_filepaths = []
        self.results_path = ""

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
