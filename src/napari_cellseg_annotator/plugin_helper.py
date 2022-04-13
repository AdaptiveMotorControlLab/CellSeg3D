import napari
from napari_cellseg_annotator import utils
from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton, QSizePolicy


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
    #         length = 2
    #         # loss plot
    #         canvas = FigureCanvas(Figure(figsize=(10, 3)))
    #
    #         train_loss = canvas.figure.add_subplot(1, 2, 1)
    #         train_loss.set_title("Epoch average loss")
    #         self.epoch_loss_values = np.linspace(0,1,length)
    #         x = [i for i in range(len(self.epoch_loss_values))]
    #         y = self.epoch_loss_values
    #         train_loss.set_xlabel("Epoch")
    #         train_loss.set_ylabel("Loss")
    #         train_loss.plot(x, y)
    #         # start, end = x[0], x[-1]
    #         # train_loss.xaxis.set_ticks(np.arange(start, end, len(x)/10))
    #         train_loss.ticklabel_format(
    #             axis="y", style="sci", scilimits=(-5, 0)
    #         )
    #
    #         bckgrd_color = (0, 0, 0, 0)  # '#262930'
    #         # dice metric validation plot
    #         dice_metric = canvas.figure.add_subplot(1, 2, 2)
    #         dice_metric.set_title("Validation metric : Mean Dice coefficient")
    #         print(int(length/2))
    #         self.metric_values = np.arange(0,1,2/length)
    #         x = np.arange(0, int(length/2), 1)
    #         y = self.metric_values
    #
    #
    #         epoch_min = (np.argmax(y) + 1) * 2
    #         dice_min = np.max(y)
    #
    #         dice_metric.set_xlabel("Epoch")
    #         dice_metric.plot(x, y)
    #
    #         # print(epoch_min)
    #         # print(dice_min)
    #         dice_metric.scatter(
    #             epoch_min, dice_min, c="r", label="Maximum Dice coeff."
    #         )
    #         dice_metric.legend(facecolor="#262930")
    #         # start, end = x[0], x[-1]
    #         # dice_metric.xaxis.set_ticks(np.arange(start, end, len(x) / 5))
    #         dice_metric.ticklabel_format(
    #             axis="y", style="sci", scilimits=(-5, 0)
    #         )
    #
    #         canvas.figure.set_facecolor(bckgrd_color)
    #         dice_metric.set_facecolor(bckgrd_color)
    #         train_loss.set_facecolor(bckgrd_color)
    #
    #         # canvas.figure.tight_layout()
    #
    #         canvas.figure.subplots_adjust(
    #             left=0.1, bottom=0.2, right=0.95, top=0.9, wspace=0.2, hspace=0
    #         )
    #
    #     canvas.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
    #
    #     # tab_index = self.addTab(canvas, "Loss plot")
    #     # self.setCurrentIndex(tab_index)
    #     self._viewer.window.add_dock_widget(canvas, area="bottom")
