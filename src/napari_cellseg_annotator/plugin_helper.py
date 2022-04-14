import napari
from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton

from napari_cellseg_annotator import utils


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

    # def plot_loss(self):
    #
    #     from matplotlib.backends.backend_qt5agg import (
    #         FigureCanvasQTAgg as FigureCanvas,
    #     )
    #     from matplotlib.figure import Figure
    #     import matplotlib.pyplot as plt
    #     import numpy as np
    #
    #     with plt.style.context("dark_background"):
    #
    #         length = 50
    #         # loss plot
    #         self.canvas = FigureCanvas(Figure(figsize=(10, 3)))
    #
    #         self.train_loss = self.canvas.figure.add_subplot(1, 2, 1)
    #         self.train_loss.set_title("Epoch average loss")
    #         self.epoch_loss_values = np.random.rand(length)
    #         x = [i for i in range(len(self.epoch_loss_values))]
    #         y = self.epoch_loss_values
    #         self.train_loss.set_xlabel("Epoch")
    #         self.train_loss.set_ylabel("Loss")
    #         self.train_loss.plot(x, y)
    #         # start, end = x[0], x[-1]
    #         # self.train_loss.xaxis.set_ticks(np.arange(start, end, len(x)/10))
    #         self.train_loss.ticklabel_format(
    #             axis="y", style="sci", scilimits=(-5, 0)
    #         )
    #
    #         bckgrd_color = (0, 0, 0, 0)  # '#262930'
    #         # dice metric validation plot
    #         self.dice_metric = self.canvas.figure.add_subplot(1, 2, 2)
    #         self.dice_metric.set_title(
    #             "Validation metric : Mean Dice coefficient"
    #         )
    #         print(int(length / 2))
    #         self.metric_values = np.random.rand(int(length / 2))
    #         x = np.linspace(0, length, int(length / 2))
    #         y = self.metric_values
    #
    #         epoch_min = (np.argmax(y) + 1) * 2
    #         dice_min = np.max(y)
    #
    #         self.dice_metric.set_xlabel("Epoch")
    #         self.dice_metric.plot(x, y)
    #
    #         # print(epoch_min)
    #         # print(dice_min)
    #         self.dice_metric.scatter(
    #             epoch_min, dice_min, c="r", label="Maximum Dice coeff."
    #         )
    #         self.dice_metric.legend(facecolor="#262930")
    #         # start, end = x[0], x[-1]
    #         # dice_metric.xaxis.set_ticks(np.arange(start, end, len(x) / 5))
    #         self.dice_metric.ticklabel_format(
    #             axis="y", style="sci", scilimits=(-5, 0)
    #         )
    #
    #         self.canvas.figure.set_facecolor(bckgrd_color)
    #         self.dice_metric.set_facecolor(bckgrd_color)
    #         self.train_loss.set_facecolor(bckgrd_color)
    #
    #         # self.canvas.figure.tight_layout()
    #
    #         self.canvas.figure.subplots_adjust(
    #             left=0.1, bottom=0.2, right=0.95, top=0.9, wspace=0.2, hspace=0
    #         )
    #
    #     self.canvas.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
    #
    #     # tab_index = self.addTab(self.canvas, "Loss plot")
    #     # self.setCurrentIndex(tab_index)
    #     self._viewer.window.add_dock_widget(self.canvas, area="bottom")
