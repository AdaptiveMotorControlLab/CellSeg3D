import os
import warnings
from datetime import datetime
from pathlib import Path

import napari
import pandas as pd
# Qt
from qtpy.QtWidgets import QVBoxLayout
from qtpy.QtWidgets import QWidget

from napari_cellseg3d import interface as ui
from napari_cellseg3d import utils

GUI_MAXIMUM_WIDTH = 225
GUI_MAXIMUM_HEIGHT = 350
GUI_MINIMUM_HEIGHT = 300


"""
plugin_dock.py
====================================
Definition of Datamanager widget, for saving labels status in csv file
"""


class Datamanager(QWidget):
    """A widget with a single checkbox that allows to store the status of
    a slice in csv file (checked/not checked)

    """

    def __init__(self, parent: "napari.viewer.Viewer"):
        """Creates the datamanager widget in the specified viewer window.

        Args:
            parent (napari.viewer.Viewer): napari Viewer for the widget to be displayed in"""

        super().__init__()

        layout = QVBoxLayout()
        self.viewer = parent
        """napari.viewer.Viewer: viewer in which the widget is displayed"""

        # add some buttons
        self.button = ui.make_button(
            "1", self.button_func, parent=self, fixed=False
        )
        self.time_label = ui.make_label("", self)
        self.time_label.setVisible(False)

        io_panel, io_layout = ui.make_container(vertical=False)
        io_layout.addWidget(
            ui.combine_blocks(
                left_or_above=self.button,
                right_or_below=self.time_label,
                horizontal=True,
            )
        )  # , alignment=utils.ABS_AL)
        io_panel.setLayout(io_layout)
        io_panel.setMaximumWidth(GUI_MAXIMUM_WIDTH)
        layout.addWidget(io_panel, alignment=ui.ABS_AL)

        self.setLayout(layout)
        # self.setMaximumHeight(GUI_MAXIMUM_HEIGHT)
        # self.setMaximumWidth(GUI_MAXIMUM_WIDTH)

        self.df = ""
        self.csv_path = ""
        self.slice_num = 0
        self.filetype = ""
        self.filename = None
        self.image_dims = self.viewer.layers[0].data.shape
        self.as_folder = False
        """Whether to load as folder or single file"""

        self.start_time = datetime.now()
        self.time = None

    def prepare(self, label_dir, filetype, model_type, checkbox, as_folder):
        """Initialize the Datamanager, which loads the csv file and updates it
        with the index of the current slice.

        Args:
        label_dir (str): label path
        filetype (str) : file extension
        model_type (str): model type
        checkbox (bool): create new dataset or not
        as_folder (bool) : load as folder or as single file
        """

        # label_dir = os.path.dirname(label_dir)
        print("csv path try :")
        print(label_dir)
        self.filetype = filetype
        self.as_folder = as_folder

        if not self.as_folder:
            self.filename = os.path.split(label_dir)[1]
            label_dir = os.path.split(label_dir)[0]
            print("Loading single image")
            print(self.filename)
            print(label_dir)

        self.df, self.csv_path = self.load_csv(label_dir, model_type, checkbox)

        print(self.csv_path, checkbox)
        # print(self.viewer.dims.current_step[0])
        self.update(self.viewer.dims.current_step[0])

    def load_csv(self, label_dir, model_type, checkbox):
        """
        Loads newest csv or create new csv

        Args:
            label_dir (str): label path
            model_type (str): model type
            checkbox ( bool ): create new dataset or not

        Returns:
            (pandas.DataFrame, str) dataframe、csv path
        """
        # if not self.as_folder :
        #     label_dir = os.path.dirname(label_dir)
        print("label dir")
        print(label_dir)
        csvs = sorted(list(Path(label_dir).glob(f"{model_type}*.csv")))
        if len(csvs) == 0:
            df, csv_path = self.create(
                label_dir, model_type
            )  # df,  train_data_dir, ...
        else:
            csv_path = str(csvs[-1])
            df = pd.read_csv(csv_path, index_col=0)
            if checkbox is True:
                csv_path = (
                    csv_path.split("_train")[0]
                    + "_train"
                    + str(
                        int(os.path.splitext(csv_path.split("_train")[1])[0])
                        + 1
                    )
                    + ".csv"
                )  # adds 1 to current csv name number
                df.to_csv(csv_path)
            else:
                pass
        self.start_time = datetime.now()
        self.time = self.start_time
        return df, csv_path

    def create(self, label_dir, model_type, filename=None):
        """
        Create a new dataframe and save the csv
        Args:
          label_dir (str): label path
          model_type (str): model type
        Returns:
         (pandas.DataFrame, str): dataframe, csv path
        """

        if self.as_folder:
            labels = sorted(
                list(
                    path.name
                    for path in Path(label_dir).glob("./*" + self.filetype)
                )
            )
        else:
            # print(self.image_dims[0])
            if self.filename is not None:
                filename = self.filename
            else:
                filename = "image"
            labels = [str(filename) for i in range(self.image_dims[0])]

        df = pd.DataFrame(
            {
                "filename": labels,
                "train": ["Not checked"] * len(labels),
                "time": [""] * len(labels),
            }
        )
        csv_path = os.path.join(label_dir, f"{model_type}_train0.csv")
        print("csv path for create")
        print(csv_path)
        df.to_csv(csv_path)

        return df, csv_path

    def update(self, slice_num):
        """Updates the Datamanager with the index of the current slice, and updates
        the text with the status contained in the csv (e.g. checked/not checked).

        Args:
            slice_num (int): index of the current slice

        """
        self.slice_num = slice_num

        print(f"New slice review started at {utils.get_time()}")
        # print(self.df)
        if len(self.df) > 1:
            self.button.setText(
                self.df.at[self.df.index[self.slice_num], "train"]
            )  # puts  button values at value of 1st csv item

        self.time = datetime.now()

        if self.button.text() == "Not checked":
            self.time_label.setVisible(False)
        else:
            self.time_label.setVisible(True)
            self.time_label.setText(
                f"Previously completed in {self.df.at[self.df.index[self.slice_num], 'time']}"
            )

    def button_func(self):  # updates csv every time you press button...
        if self.viewer.dims.ndisplay != 2:
            # TODO test if undefined behaviour or if okay
            warnings.warn("Please switch back to 2D mode !")
            return
        if self.button.text() == "Not checked":

            start_time = self.time
            finish_time = datetime.now()
            self.time = finish_time

            time_diff = utils.time_difference(start_time, finish_time)
            print(f"Time taken : {time_diff}")

            self.button.setText("Checked")
            self.time_label.setVisible(True)
            self.time_label.setText(f"Completed in {time_diff}")
            self.df.at[self.df.index[self.slice_num], "train"] = "Checked"
            self.df.at[self.df.index[self.slice_num], "time"] = time_diff
            self.df.to_csv(self.csv_path)

        else:
            self.button.setText("Not checked")
            self.time_label.setVisible(False)
            self.time = datetime.now()
            self.df.at[self.df.index[self.slice_num], "train"] = "Not checked"
            self.df.to_csv(self.csv_path)

    # def move_data(self):
    #     shutil.copy(
    #         self.df.at[self.df.index[self.slice_num], "filename"],
    #         self.train_data_dir,
    #     )
    #
    # def delete_data(self):
    #     os.remove(
    #         os.path.join(
    #             self.train_data_dir,
    #             os.path.basename(
    #                 self.df.at[self.df.index[self.slice_num], "filename"]
    #             ),
    #         )
    #     )
    #
    # def check_all_data_and_mod(self):
    #     for i in range(len(self.df)):
    #         if self.df.at[self.df.index[i], "train"] == "Checked":
    #             try:
    #                 shutil.copy(
    #                     self.df.at[self.df.index[i], "filename"],
    #                     self.train_data_dir,
    #                 )
    #             except:
    #                 pass
    #         else:
    #             try:
    #                 os.remove(
    #                     os.path.join(
    #                         self.train_data_dir,
    #                         os.path.basename(
    #                             self.df.at[self.df.index[i], "filename"]
    #                         ),
    #                     )
    #                 )
    #             except:
    #                 pass
