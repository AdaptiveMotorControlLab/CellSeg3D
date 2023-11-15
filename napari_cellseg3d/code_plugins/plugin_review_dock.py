"""Widget opened when a new Review session is started."""
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import napari

# Qt
from qtpy.QtWidgets import QVBoxLayout, QWidget

from napari_cellseg3d import interface as ui
from napari_cellseg3d import utils

GUI_MAXIMUM_WIDTH = 225
GUI_MAXIMUM_HEIGHT = 350
GUI_MINIMUM_HEIGHT = 300
TIMER_FORMAT = "%H:%M:%S"

logger = utils.LOGGER
"""
plugin_dock.py
====================================
Definition of Datamanager widget, for saving labels status in csv file
"""


class Datamanager(QWidget):
    """A widget with a single checkbox that allows to store the status of a slice in csv file (checked/not checked)."""

    def __init__(self, parent: "napari.viewer.Viewer"):
        """Creates the datamanager widget in the specified viewer window.

        Args:
            parent (napari.viewer.Viewer): napari Viewer for the widget to be displayed in
        """
        super().__init__()

        layout = QVBoxLayout()
        self.viewer = parent
        """napari.viewer.Viewer: viewer in which the widget is displayed"""

        # add some buttons
        self.button = ui.Button(
            "1", self._button_func, parent=self, fixed=True
        )
        self.time_label = ui.make_label("", self)
        self.time_label.setVisible(False)

        self.pause_box = ui.CheckBox(
            "Pause timer", self.pause_timer, parent=self, fixed=True
        )

        io_panel = ui.ContainerWidget()
        io_layout = io_panel.layout
        io_layout.addWidget(self.button, alignment=ui.ABS_AL)
        io_layout.addWidget(
            ui.combine_blocks(
                left_or_above=self.pause_box,
                right_or_below=self.time_label,
                horizontal=True,
            ),
            alignment=ui.ABS_AL,
        )

        io_panel.setLayout(io_layout)
        io_panel.setMaximumWidth(GUI_MAXIMUM_WIDTH)
        layout.addWidget(io_panel, alignment=ui.ABS_AL)

        self.setLayout(layout)
        # self.setMaximumHeight(GUI_MAXIMUM_HEIGHT)
        # self.setMaximumWidth(GUI_MAXIMUM_WIDTH)

        self.df = None
        self.csv_path = None
        self.slice_num = 0
        self.filetype = ""
        self.filename = None
        self.image_dims = self.viewer.layers[0].data.shape
        self.as_folder = False
        """Whether to load as folder or single file"""

        self.start_time = datetime.now()
        self.time_elapsed = timedelta()
        self.pause_start = None
        self.is_paused = False
        # self.pause_time = None

    def pause_timer(self):
        """Pause the timer for the review time."""
        if self.pause_box.isChecked():
            self.time_label.setVisible(True)

            self.pause_start = datetime.now()
            self.time_elapsed += self.pause_start - self.start_time
            self.pause_box.setText("Resume timer")
            self.time_label.setText(
                f"({utils.time_difference(timedelta(),self.time_elapsed)})"
            )
            self.is_paused = True
        else:
            self.time_label.setVisible(False)
            self.pause_box.setText("Pause timer")

            # self.pause_time = datetime.now() - self.pause_start
            self.start_time = datetime.now()
            self.is_paused = False
        self._update_time_csv()

    def _update_time_csv(self):
        if not self.is_paused:
            self.time_elapsed += datetime.now() - self.start_time
            self.start_time = datetime.now()
        str_time = utils.time_difference(timedelta(), self.time_elapsed)
        logger.info(f"Time elapsed : {str_time}")
        self.df.at[0, "time"] = str_time
        self.df.to_csv(self.csv_path)

    def prepare(self, label_dir, filetype, model_type, checkbox):
        """Initialize the Datamanager, which loads the csv file and updates it with the index of the current slice.

        Args:
            label_dir (str): label path
            filetype (str) : file extension
            model_type (str): model type
            checkbox (bool): create new dataset or not
            as_folder (bool) : load as folder or as single file
        """
        # label_dir = os.path.dirname(label_dir)
        logger.debug("csv path try :")
        logger.debug(label_dir)
        self.filetype = filetype

        if not self.as_folder:
            p = Path(label_dir)
            self.filename = p.name
            label_dir = p.parents[0]
            logger.info(f"Loading single image : {self.filename}")
            logger.debug(label_dir)

        self.df, self.csv_path = self.load_csv(label_dir, model_type, checkbox)

        logger.debug(f"csv path : {self.csv_path}")
        logger.debug(f"Create new dataset : {checkbox}")
        # logger.debug(self.viewer.dims.current_step[0])
        self.update_dm(self.viewer.dims.current_step[0])

    def load_csv(self, label_dir, model_type, checkbox):
        """Loads newest csv or create new csv.

        Args:
            label_dir (str): label path
            model_type (str): model type
            checkbox ( bool ): create new dataset or not

        Returns:
            (pandas.DataFrame, str) dataframe、csv path
        """
        # if not self.as_folder :
        #     label_dir = os.path.dirname(label_dir)
        logger.debug("label dir")
        logger.debug(label_dir)
        csvs = sorted(list(Path(str(label_dir)).glob(f"{model_type}*.csv")))
        if len(csvs) == 0:
            df, csv_path = self.create_csv(
                label_dir, model_type
            )  # df,  train_data_dir, ...
        else:
            csv_path = str(csvs[-1])
            df = pd.read_csv(csv_path, index_col=0)
            if checkbox is True:
                csv_path = (
                    csv_path.split("_train")[0]
                    + "_train"
                    + str(int(Path(csv_path.split("_train")[1]).parent) + 1)
                    + ".csv"
                )  # adds 1 to current csv name number
                df.to_csv(csv_path)
            else:
                pass

        recorded_time = df.at[0, "time"]
        # logger.debug("csv load time")
        # logger.debug(recorded_time)
        t = datetime.strptime(recorded_time, TIMER_FORMAT)
        self.time_elapsed = timedelta(
            hours=t.hour, minutes=t.minute, seconds=t.second
        )
        # logger.debug(self.time_elapsed)
        return df, csv_path

    def create_csv(self, label_dir, model_type):
        """Create a new dataframe and save the csv.

        Args:
          label_dir (str): label path
          model_type (str): model type
        Returns:
         (pandas.DataFrame, str): dataframe, csv path.
        """
        if self.as_folder:
            labels = sorted(
                list(
                    path.name
                    for path in Path(str(label_dir)).glob(
                        "./*" + self.filetype
                    )
                )
            )
        else:
            # logger.debug(self.image_dims[0])
            filename = self.filename if self.filename is not None else "image"
            labels = [str(filename) for i in range(self.image_dims[0])]

        df = pd.DataFrame(
            {
                "filename": labels,
                "train": ["Not checked"] * len(labels),
                "time": [""] * len(labels),
            }
        )
        df.at[0, "time"] = "00:00:00"

        csv_path = Path(label_dir) / Path(f"{model_type}_train0.csv")
        if not csv_path.parent.exists():
            csv_path.parent.mkdir(parents=True)
        logger.debug(f"CSV path : {csv_path}")
        df.to_csv(str(csv_path))

        return df, csv_path

    def _update_button(self):
        if len(self.df) > 1:
            self.button.setText(
                self.df.at[self.df.index[self.slice_num], "train"]
            )  # puts  button values at value of 1st csv item

    def update_dm(self, slice_num):
        """Updates the Datamanager with the index of the current slice.

        Also updates the text with the status contained in the csv (e.g. checked/not checked).

        Args:
            slice_num (int): index of the current slice
        """
        self.slice_num = slice_num
        self._update_time_csv()

        logger.info(f"New slice review started at {utils.get_time()}")
        # logger.debug(self.df)

        try:
            self._update_button()
        except IndexError:
            self.slice_num -= 1
            self._update_button()

    def _button_func(self):  # updates csv every time you press button...
        if self.viewer.dims.ndisplay != 2:
            # TODO test if undefined behaviour or if okay
            logger.warning("Please switch back to 2D mode !")
            return

        self._update_time_csv()

        if self.button.text() == "Not checked":
            self.button.setText("Checked")
            self.df.at[self.df.index[self.slice_num], "train"] = "Checked"
            self.df.to_csv(self.csv_path)
        else:
            self.button.setText("Not checked")
            self.df.at[self.df.index[self.slice_num], "train"] = "Not checked"
            self.df.to_csv(self.csv_path)

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
