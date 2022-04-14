import napari
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QSpinBox,
    QSizePolicy,
)

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

        self.test = True

        if self.test:
            self.epoch = QSpinBox()
            self.epoch.setValue(0)
            self.epoch.setSingleStep(2)
            self.epoch.valueChanged.connect(self.update_loss_plot)

        self.build()
        # self.update_loss_plot()

    def build(self):
        vbox = QVBoxLayout()
        vbox.addWidget(self.btn1)
        vbox.addWidget(self.btn2)
        vbox.addWidget(self.btnc)
        vbox.addWidget(self.epoch)
        self.setLayout(vbox)
        # self.show()
        # self._viewer.window.add_dock_widget(self, name="Help/About...", area="right")

    def close(self):
        self._viewer.window.remove_dock_widget(self)

    ################ TESTING
    def update_loss_plot(self):
        if not self.test:
            return
        import numpy as np
        import matplotlib.pyplot as plt

        length = 50
        epoch = self.epoch.value()
        loss = np.random.rand(length)
        dice_metric = np.random.rand(int(length / 2))

        # print("plot upd")
        # print(epoch)
        # print(loss)
        # print(dice_metric)
        if epoch < 4:
            return
        elif epoch == 4:

            from matplotlib.backends.backend_qt5agg import (
                FigureCanvasQTAgg as FigureCanvas,
            )
            from matplotlib.figure import Figure
            import matplotlib.pyplot as plt
            import numpy as np

            bckgrd_color = (0, 0, 0, 0)  # '#262930'
            with plt.style.context("dark_background"):

                self.canvas = FigureCanvas(Figure(figsize=(10, 3)))
                # loss plot
                self.train_loss_plot = self.canvas.figure.add_subplot(1, 2, 1)
                # dice metric validation plot
                self.dice_metric_plot = self.canvas.figure.add_subplot(1, 2, 2)

                self.canvas.figure.set_facecolor(bckgrd_color)
                self.dice_metric_plot.set_facecolor(bckgrd_color)
                self.train_loss_plot.set_facecolor(bckgrd_color)

                # self.canvas.figure.tight_layout()

                self.canvas.figure.subplots_adjust(
                    left=0.1,
                    bottom=0.2,
                    right=0.95,
                    top=0.9,
                    wspace=0.2,
                    hspace=0,
                )

            self.canvas.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

            # tab_index = self.addTab(self.canvas, "Loss plot")
            # self.setCurrentIndex(tab_index)
            self._viewer.window.add_dock_widget(self.canvas, area="bottom")
            self.plot_loss(loss, dice_metric)
        else:
            with plt.style.context("dark_background"):

                self.train_loss_plot.cla()
                self.dice_metric_plot.cla()

                self.plot_loss(loss, dice_metric)

    def plot_loss(self, loss, dice_metric):
        import numpy as np

        if not self.test:
            return
        self.val_interval = 2

        # update loss
        self.train_loss_plot.set_title("Epoch average loss")
        self.train_loss_plot.set_xlabel("Epoch")
        self.train_loss_plot.set_ylabel("Loss")
        x = [i + 1 for i in range(len(loss))]
        y = loss
        self.train_loss_plot.plot(x, y)
        # update metrics
        x = [self.val_interval * (i + 1) for i in range(len(dice_metric))]
        y = dice_metric

        epoch_min = (np.argmax(y) + 1) * self.val_interval
        dice_min = np.max(y)

        self.dice_metric_plot.plot(x, y)
        self.dice_metric_plot.set_title(
            "Validation metric : Mean Dice coefficient"
        )
        self.dice_metric_plot.set_xlabel("Epoch")

        self.dice_metric_plot.scatter(
            epoch_min, dice_min, c="r", label="Maximum Dice coeff."
        )
        self.dice_metric_plot.legend(facecolor="#262930", loc="upper right")
        self.canvas.draw_idle()
