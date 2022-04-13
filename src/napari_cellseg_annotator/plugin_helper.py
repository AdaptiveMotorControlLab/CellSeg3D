import napari
from napari_cellseg_annotator import utils
from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton, QSizePolicy

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


class Helper(QWidget):
    # widget testing
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        # self.master = parent
        self.help_url = (
            "https://github.com/C-Achard/cellseg-annotator-test/tree/main"
        )

        self.about_url = "https://wysscenter.ch/advances/3d-computer-vision-for-brain-analysis"
        self._viewer = viewer
        self.btn1 = QPushButton("Help...", self)
        # self.btn1.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn1.clicked.connect(lambda: utils.open_url(self.help_url))
        self.btn2 = QPushButton("About...", self)
        # self.btn2.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn2.clicked.connect(lambda: utils.open_url(self.about_url))
        self.btnc = QPushButton("Close", self)
        # self.btnc.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btnc.clicked.connect(self.close)
        self.build()
        self.plot_loss()


    def build(self):
        vbox = QVBoxLayout()
        vbox.addWidget(self.btn1)
        vbox.addWidget(self.btn2)
        vbox.addWidget(self.btnc)
        self.setLayout(vbox)
        # self.show()
        # self._viewer.window.add_dock_widget(self, name="Help/About...", area="right")

    def close(self):
        self._viewer.window.remove_dock_widget(self)

    def plot_loss(self):
        with plt.style.context("dark_background"):
            # loss plot
            canvas = FigureCanvas(Figure(figsize=(10, 3)))

            train_loss = canvas.figure.add_subplot(1, 2, 1)
            train_loss.set_title("Epoch Average Loss")

            # x = [i + 1 for i in range(len(self.epoch_loss_values))]
            # y = self.epoch_loss_values
            train_loss.set_xlabel("epoch")
            # train_loss.plot(x, y)
            # train_loss.set_xticks(x)
            train_loss.ticklabel_format(axis="y", style="sci", scilimits=(-5, 0))

            dice_metric = canvas.figure.add_subplot(1, 2, 2)
            dice_metric.set_title("Val Mean Dice")

            # x = [
            #     self.val_interval * (i + 1) for i in range(len(self.metric_values))
            # ]
            # y = self.metric_values
            dice_metric.set_xlabel("epoch")
            # dice_metric.plot(x, y)
            # dice_metric.set_xticks(x)
            dice_metric.ticklabel_format(axis="y", style="sci", scilimits=(-5, 0))

            bckgrd_color = '#262930'

            canvas.figure.set_facecolor(bckgrd_color)
            dice_metric.set_facecolor(bckgrd_color)
            train_loss.set_facecolor(bckgrd_color)

            # canvas.figure.tight_layout()
            canvas.figure.subplots_adjust(
                left=0.1, bottom=0.2, right=0.95, top=0.9, wspace=0.2, hspace=0
            )

        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)

        # tab_index = self.addTab(canvas, "Loss plot")
        # self.setCurrentIndex(tab_index)
        self._viewer.window.add_dock_widget(canvas, area="bottom")