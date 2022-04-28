import napari

# Qt
from qtpy.QtWidgets import QSizePolicy
from qtpy.QtWidgets import QSpinBox
from qtpy.QtWidgets import QVBoxLayout
from qtpy.QtWidgets import QWidget

# local
from napari_cellseg_annotator import utils
from napari_cellseg_annotator import interface as ui


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
        self.btn1 = ui.make_button(
            "Help...", lambda: ui.open_url(self.help_url)
        )
        self.btn2 = ui.make_button(
            "About...", lambda: ui.open_url(self.about_url)
        )
        self.btnc = ui.make_button("Close", self.close)

        ###################
        ###################
        ###################
        ###################
        # TODO test remove later
        self.test = utils.ENABLE_TEST_MODE()

        if self.test:
            self.dock = None

            self.epoch = ui.make_n_spinboxes(1, 0, 1000, step=2)
            self.epoch.setValue(0)
            self.epoch.setRange(0, 1000)
            self.epoch.setSingleStep(2)
            self.epoch.valueChanged.connect(self.update_loss_plot)
        ###################
        ###################
        ###################
        ###################

        self.build()

    def build(self):
        vbox = QVBoxLayout()
        vbox.addWidget(self.btn1, alignment=ui.LEFT_AL)
        vbox.addWidget(self.btn2, alignment=ui.LEFT_AL)
        vbox.addWidget(self.btnc, alignment=ui.LEFT_AL)
        if self.test:
            vbox.addWidget(self.epoch, alignment=ui.ABS_AL)
        self.setLayout(vbox)
        # self.show()
        # self._viewer.window.add_dock_widget(self, name="Help/About...", area="right")

    def close(self):
        if self.test and self.dock is not None:  # TODO remove
            self._viewer.window.remove_dock_widget(self.dock)
        self._viewer.window.remove_dock_widget(self)

    ################
    ################
    ################
    ################ TESTING
    def update_loss_plot(self):
        if not self.test:
            return
        import matplotlib.pyplot as plt
        import numpy as np

        epoch = self.epoch.value()
        length = epoch
        loss = np.random.rand(length)
        dice_metric = np.random.rand(int(length / 2))

        # print("plot upd")
        # print(epoch)
        # print(loss)
        # print(dice_metric)
        if epoch < 4:
            return
        elif epoch == 4:

            import matplotlib.pyplot as plt
            import numpy as np
            from matplotlib.backends.backend_qt5agg import (
                FigureCanvasQTAgg as FigureCanvas,
            )
            from matplotlib.figure import Figure

            bckgrd_color = (0, 0, 0, 0)  # '#262930'
            with plt.style.context("dark_background"):

                self.canvas = FigureCanvas(Figure(figsize=(7, 2.5)))
                # loss plot
                self.train_loss_plot = self.canvas.figure.add_subplot(1, 2, 1)
                # dice metric validation plot
                self.dice_metric_plot = self.canvas.figure.add_subplot(1, 2, 2)

                self.canvas.figure.set_facecolor(bckgrd_color)
                self.dice_metric_plot.set_facecolor(bckgrd_color)
                self.train_loss_plot.set_facecolor(bckgrd_color)

                self.canvas.figure.tight_layout()

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
            self.dock = self._viewer.window.add_dock_widget(
                self.canvas, area="bottom"
            )
            self.plot_loss(loss, dice_metric)
        else:
            with plt.style.context("dark_background"):

                self.train_loss_plot.cla()
                self.dice_metric_plot.cla()

                self.plot_loss(loss, dice_metric)

    def plot_loss(self, loss, dice_metric):
        import matplotlib.pyplot as plt
        import numpy as np

        if not self.test:
            return
        self.val_interval = 2

        # update loss
        with plt.style.context("dark_background"):
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

            self.dice_metric_plot.plot(x, y, zorder=1)
            self.dice_metric_plot.set_title(
                "Validation metric : Mean Dice coefficient"
            )
            self.dice_metric_plot.set_xlabel("Epoch")
            self.dice_metric_plot.set_ylabel("Dice")

            self.dice_metric_plot.scatter(
                epoch_min,
                dice_min,
                c="r",
                label="Maximum Dice coeff.",
                zorder=5,
            )
            self.dice_metric_plot.legend(
                facecolor="#262930", loc="lower right"
            )
            self.canvas.draw_idle()
            ##########################
            ##########################
            ##########################
            ########################## END of testing
