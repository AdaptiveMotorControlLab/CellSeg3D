import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.figure import Figure
from monai.transforms import SpatialPad, ToTensor
from tifffile import imread

from napari_cellseg3d import interface as ui
from napari_cellseg3d import utils
from napari_cellseg3d.model_instance_seg import to_semantic
from napari_cellseg3d.plugin_base import BasePluginFolder


class MetricsUtils(BasePluginFolder):
    """Plugin to evaluate metrics between two sets of labels, ground truh and prediction"""

    def __init__(self, viewer: "napari.viewer.Viewer", parent):
        """Creates a MetricsUtils widget for computing and plotting dice metrics between labels.
        Args:
            viewer: viewer to display the widget in
            parent : parent widget
        """
        super().__init__(viewer, parent)

        self._viewer = viewer
        """Viewer to display widget in"""

        self.layout = None
        """Called for plotting"""
        self.canvas = None
        """Canvas to render plots on"""
        self.plots = []
        """Array that references all plots currently on the window"""

        ######################################
        # interface
        self.btn_compute_dice = ui.make_button(
            "Compute Dice", self.compute_dice
        )
        self.btn_reset_plot = ui.make_button("Clear plots", self.remove_plots)

        self.btn_result_path.setVisible(False)
        self.lbl_result_path.setVisible(False)

        self.build()

        ######################################
        # TODO test remove
        import glob
        import os

        if utils.ENABLE_TEST_MODE():
            ground_directory = "C:/Users/Cyril/Desktop/Proj_bachelor/data/cropped_visual/train/lab"
            # ground_directory = "C:/Users/Cyril/Desktop/test/labels"
            pred_directory = "C:/Users/Cyril/Desktop/test/pred"
            # pred_directory = "C:/Users/Cyril/Desktop/test"
            self.images_filepaths = sorted(
                glob.glob(os.path.join(ground_directory, "*.tif"))
            )
            self.labels_filepaths = sorted(
                glob.glob(os.path.join(pred_directory, "*.tif"))
            )
        ###############################################################################

    def build(self):
        """Builds the layout of the widget."""

        self.lbl_filetype.setVisible(False)

        w, self.layout = ui.make_container_widget()

        metrics_group_w, metrics_group_l = ui.make_group("Metrics")

        self.lbl_image_files.setText("Ground truth")

        metrics_group_l.addWidget(
            ui.combine_blocks(
                second=self.btn_image_files,
                first=self.lbl_image_files,
                min_spacing=70,
            ),
            alignment=ui.LEFT_AL,
        )

        self.lbl_label_files.setText("Prediction")

        metrics_group_l.addWidget(
            ui.combine_blocks(
                second=self.btn_label_files,
                first=self.lbl_label_files,
                min_spacing=70,
            ),
            alignment=ui.LEFT_AL,
        )

        metrics_group_l.addWidget(self.btn_compute_dice, alignment=ui.LEFT_AL)

        metrics_group_w.setLayout(metrics_group_l)

        self.layout.addWidget(metrics_group_w)

        self.layout.addWidget(self.make_close_button(), alignment=ui.LEFT_AL)

        self.layout.addWidget(self.btn_reset_plot, alignment=ui.LEFT_AL)
        self.btn_reset_plot.setVisible(False)

        ui.make_scrollable(self.layout, self)

    def plot_dice(self, dice_coeffs):
        """Plots the dice loss for each pair of labels on viewer"""
        self.btn_reset_plot.setVisible(True)
        colors = []

        bckgrd_color = (0, 0, 0, 0)

        for coeff in dice_coeffs:  # TODO add threshold manual setting
            if coeff < 0.5:
                colors.append(ui.dark_red)  # 72071d # crimson red
            else:
                colors.append(ui.default_cyan)  # 8dd3c7 # turquoise cyan
        with plt.style.context("dark_background"):
            if self.canvas is None:
                self.canvas = FigureCanvas(Figure(figsize=(2, 5)))
                self.layout.addWidget(self.canvas)
                self.canvas.figure.tight_layout()
            else:
                self.dice_plot.cla()
            self.canvas.figure.set_facecolor(bckgrd_color)
            dice_plot = self.canvas.figure.add_subplot(1, 1, 1)
            labels = np.array(range(len(dice_coeffs))) + 1

            dice_plot.barh(labels, dice_coeffs, color=colors)
            dice_plot.set_facecolor(bckgrd_color)

            dice_plot.invert_yaxis()

            self.plots.append(self.canvas)
            dice_plot.axvline(0.5, color=ui.dark_red)
            dice_plot.set_title(
                f"Session {len(self.plots)}\nMean dice : {np.mean(dice_coeffs):.4f}"
            )
            # dice_plot.set_xticks(rotation=45)
            dice_plot.set_xlabel(f"Dice coefficient")
            # dice_plot.set_ylabel("Labels pair id", rotation=90)

            self.canvas.draw_idle()

    def remove_plots(self):
        """Clears plots from window view"""
        if len(self.plots) != 0:
            for p in self.plots:
                p.setVisible(False)
        self.plots = []
        self.btn_reset_plot.setVisible(False)

    def compute_dice(self):
        """Computes the dice metric between pairs of labels. Rotates the prediction label to find matching orientation as well."""
        # u = 0
        # t = 0
        total_metrics = []
        self.canvas = (
            None  # kind of unsafe way to stack plots... but it works.
        )
        id = 0
        for ground_path, pred_path in zip(
            self.images_filepaths, self.labels_filepaths
        ):
            id += 1
            ground = imread(ground_path)
            pred = imread(pred_path)

            ground = to_semantic(ground).astype(np.int8)
            pred = to_semantic(pred).astype(np.int8)

            pred_dims = pred.shape[-3:]
            # ground_dims = ground.shape[-3:]
            # print(pred_dims)
            # print(ground_dims)
            pad_pred = utils.get_padding_dim(pred_dims)
            # pad_ground = utils.get_padding_dim(ground_dims)

            # origin, target = utils.align_array_sizes(array_shape=pad_ground, target_shape=pad_pred)

            # ground = np.moveaxis(ground, origin, target)
            print(ground.shape)
            print(pred.shape)

            while len(pred.shape) < 5:
                pred = np.expand_dims(pred, axis=0)
                # print("-")
            while len(ground.shape) < 4:
                ground = np.expand_dims(ground, axis=0)
            ground = (SpatialPad(pad_pred)(ToTensor()(ground))).numpy()
            while len(ground.shape) < len(pred.shape):
                ground = np.expand_dims(ground, axis=0)
                # print("&")

            # print(ground.shape)
            # print(pred.shape)

            if ground.shape != pred.shape:
                raise ValueError(
                    f"Padded sizes of images do not match ! Padded ground label : {ground.shape} Padded pred label : {pred.shape}"
                )
            # if u < 1:
            # self._viewer.add_image(
            #     ground, name="ground", colormap="blue", opacity=0.7
            # )
            # self._viewer.add_image(pred, name="pred", colormap="red")
            # self._viewer.add_image(
            #     np.rot90(pred[0][0], axes=(0, 1)),
            #     name="pred flip 0",
            #     colormap="red",
            #     opacity=0.7,
            # )
            # self._viewer.add_image(
            #     np.rot90(pred[0][0], axes=(1, 2)),
            #     name="pred flip 1",
            #     colormap="red",
            #     opacity=0.7,
            # )
            # self._viewer.add_image(
            #     np.rot90(pred[0][0], axes=(0, 2)),
            #     name="pred flip 2",
            #     colormap="red",
            #     opacity=0.7,
            # )
            # u += 1

            # TODO add rotation toggle
            pred_flip_x = np.rot90(pred[0][0], axes=(0, 1))
            pred_flip_y = np.rot90(pred[0][0], axes=(1, 2))
            pred_flip_z = np.rot90(pred[0][0], axes=(0, 2))
            scores = []

            for p in [pred[0][0], pred_flip_x, pred_flip_y, pred_flip_z]:
                scores.append(utils.dice_coeff(p, ground))
                scores.append(utils.dice_coeff(np.flip(p), ground))
                for i in range(3):
                    scores.append(utils.dice_coeff(np.flip(p, axis=i), ground))

            # if t < 1:
            #     for i in range(3):
            #         self._viewer.add_image(
            #             np.flip(pred_flip_x, axis=i),
            #             name=f"flip",
            #             colormap="green",
            #             opacity=0.7,
            #         )
            #     t += 1

            # print(scores)
            score = max(scores)
            if score < 0.5:
                # TODO add filename ?
                self._viewer.dims.ndisplay = 3
                self._viewer.add_image(
                    ground, name=f"ground_{i+1}", colormap="blue", opacity=0.7
                )
                self._viewer.add_image(
                    pred, name=f"pred_{i+1}", colormap="red", opacity=0.7
                )
            total_metrics.append(score)
        print(f"DICE METRIC :{total_metrics}")
        self.plot_dice(total_metrics)
