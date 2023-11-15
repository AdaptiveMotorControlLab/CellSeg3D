"""CRF plugin for napari_cellseg3d."""
import contextlib
from functools import partial
from pathlib import Path

import napari.layers
from qtpy.QtWidgets import QSizePolicy
from tqdm import tqdm

from napari_cellseg3d import config, utils
from napari_cellseg3d import interface as ui
from napari_cellseg3d.code_models.crf import (
    CRF_INSTALLED,
    CRFWorker,
    crf_with_config,
)
from napari_cellseg3d.code_plugins.plugin_base import BasePluginUtils
from napari_cellseg3d.utils import LOGGER as logger


# TODO add CRF on folder
class CRFParamsWidget(ui.GroupedWidget):
    """Use this widget when adding the crf as part of another widget (rather than a standalone widget)."""

    def __init__(self, parent=None):
        """Create a widget to set CRF parameters."""
        super().__init__(title="CRF parameters", parent=parent)
        #######
        # CRF params #
        self.sa_choice = ui.DoubleIncrementCounter(
            default=10, parent=self, text_label="Alpha std"
        )
        self.sb_choice = ui.DoubleIncrementCounter(
            default=5, parent=self, text_label="Beta std"
        )
        self.sg_choice = ui.DoubleIncrementCounter(
            default=1, parent=self, text_label="Gamma std"
        )
        self.w1_choice = ui.DoubleIncrementCounter(
            default=10, parent=self, text_label="Weight appearance"
        )
        self.w2_choice = ui.DoubleIncrementCounter(
            default=5, parent=self, text_label="Weight smoothness"
        )
        self.n_iter_choice = ui.IntIncrementCounter(
            default=5, parent=self, text_label="Number of iterations"
        )
        #######
        self._build()
        self._set_tooltips()

    def _build(self):
        if not CRF_INSTALLED:
            ui.add_widgets(
                self.layout,
                [
                    ui.make_label(
                        "ERROR: CRF not installed.\nPlease refer to the documentation to install it."
                    ),
                ],
            )
            self._set_layout()
            return
        ui.add_widgets(
            self.layout,
            [
                # self.sa_choice.label,
                self.sa_choice,
                # self.sb_choice.label,
                self.sb_choice,
                # self.sg_choice.label,
                self.sg_choice,
                # self.w1_choice.label,
                self.w1_choice,
                # self.w2_choice.label,
                self.w2_choice,
                # self.n_iter_choice.label,
                self.n_iter_choice,
            ],
        )
        self._set_layout()

    def _set_tooltips(self):
        self.sa_choice.setToolTip(
            "SA : Standard deviation of the Gaussian kernel in the appearance term."
        )
        self.sb_choice.setToolTip(
            "SB : Standard deviation of the Gaussian kernel in the smoothness term."
        )
        self.sg_choice.setToolTip(
            "SG : Standard deviation of the Gaussian kernel in the gradient term."
        )
        self.w1_choice.setToolTip(
            "W1 : Weight of the appearance term in the CRF."
        )
        self.w2_choice.setToolTip(
            "W2 : Weight of the smoothness term in the CRF."
        )
        self.n_iter_choice.setToolTip("Number of iterations of the CRF.")

    def make_config(self):
        """Make a CRF config from the widget values."""
        return config.CRFConfig(
            sa=self.sa_choice.value(),
            sb=self.sb_choice.value(),
            sg=self.sg_choice.value(),
            w1=self.w1_choice.value(),
            w2=self.w2_choice.value(),
            n_iters=self.n_iter_choice.value(),
        )


class CRFWidget(BasePluginUtils):
    """Widget to run CRF post-processing."""

    save_path = Path.home() / "cellseg3d" / "crf"

    def __init__(self, viewer, parent=None):
        """Create a widget for CRF post-processing.

        Args:
            viewer: napari viewer to display the widget
            parent: parent widget. Defaults to None.
        """
        super().__init__(viewer, parent=parent)
        self._viewer = viewer

        self.start_button = ui.Button("Start", self._start, parent=self)
        self.crf_params_widget = CRFParamsWidget(parent=self)
        self.io_panel = self._build_io_panel()
        self.io_panel.setVisible(False)

        self.results_filewidget.setVisible(True)
        self.label_layer_loader.setVisible(True)
        self.label_layer_loader.set_layer_type(
            napari.layers.Image
        )  # to load all crf-compatible inputs, not int only
        self.image_layer_loader.setVisible(True)
        self.label_layer_loader.layer_list.label.setText("Model output :")

        if CRF_INSTALLED:
            self.start_button.setVisible(True)
        else:
            self.start_button.setVisible(False)

        self.result_layer = None
        self.result_name = None
        self.crf_results = []

        self.results_path = str(self.save_path)
        self.results_filewidget.text_field.setText(self.results_path)
        self.results_filewidget.check_ready()

        self._container = ui.ContainerWidget(parent=self, l=11, t=11, r=11)
        self.layout = self._container.layout

        self._build()

        self.worker = None
        self.log = None
        self.layer = None

    def _build(self):
        self.setMinimumWidth(100)
        ui.add_widgets(
            self.layout,
            [
                self.image_layer_loader,
                self.label_layer_loader,
                self.save_label,
                self.results_filewidget,
                ui.make_label(""),
                self.crf_params_widget,
                ui.make_label(""),
                self.start_button,
            ],
        )
        # self.io_panel.setLayout(self.io_panel.layout)
        self.setLayout(self.layout)

        ui.ScrollArea.make_scrollable(
            self.layout, self, max_wh=[ui.UTILS_MAX_WIDTH, ui.UTILS_MAX_HEIGHT]
        )
        self._container.setSizePolicy(
            QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding
        )
        return self._container

    def make_config(self):
        """Make a CRF config from the widget values."""
        return self.crf_params_widget.make_config()

    def print_config(self):
        """Print the CRF config to the logger."""
        logger.info("CRF config:")
        for item in self.make_config().__dict__.items():
            logger.info(f"{item[0]}: {item[1]}")

    def _check_ready(self):
        if len(self.label_layer_loader.layer_list) < 1:
            logger.warning("No label layer loaded")
            return False
        if len(self.image_layer_loader.layer_list) < 1:
            logger.warning("No image layer loaded")
            return False

        if len(self.label_layer_loader.layer_data().shape) < 3:
            logger.warning("Label layer must be 3D")
            return False
        if len(self.image_layer_loader.layer_data().shape) < 3:
            logger.warning("Image layer must be 3D")
            return False
        if (
            self.label_layer_loader.layer_data().shape[-3:]
            != self.image_layer_loader.layer_data().shape[-3:]
        ):
            logger.warning("Image and label layers must have the same shape!")
            return False

        return True

    def run_crf_on_batch(self, images_list: list, labels_list: list, log=None):
        """Run CRF on a batch of images and labels."""
        self.crf_results = []
        for image, label in zip(images_list, labels_list):
            tqdm(
                unit="B",
                total=len(images_list),
                position=0,
                file=log,
            )
            result = crf_with_config(image, label, self.make_config())
            self.crf_results.append(result)
        return self.crf_results

    def _prepare_worker(self, images_list: list, labels_list: list):
        """Prepare the CRF worker."""
        self.worker = CRFWorker(
            images_list=images_list,
            labels_list=labels_list,
            config=self.make_config(),
        )

        self.worker.started.connect(self._on_start)
        self.worker.yielded.connect(partial(self._on_yield))
        self.worker.errored.connect(partial(self._on_error))
        self.worker.finished.connect(self._on_finish)

    def _start(self):
        if not self._check_ready():
            return

        self.result_layer = self.label_layer_loader.layer()
        self.result_name = self.label_layer_loader.layer_name()

        utils.mkdir_from_str(self.results_path)

        image_list = [self.image_layer_loader.layer_data()]
        labels_list = [self.label_layer_loader.layer_data()]
        [logger.debug(f"Image shape: {image.shape}") for image in image_list]
        [
            logger.debug(f"Label shape: {labels.shape}")
            for labels in labels_list
        ]

        self._prepare_worker(image_list, labels_list)

        if self.worker.is_running:  # if worker is running, tries to stop
            logger.info("Stop request, waiting for previous job to finish")
            self.start_button.setText("Stopping...")
            self.worker.quit()
        else:  # once worker is started, update buttons
            self.start_button.setText("Running...")
            logger.info("Starting CRF...")
            self.worker.start()

    def _on_yield(self, result):
        self.crf_results.append(result)
        utils.save_layer(
            self.results_filewidget.text_field.text(),
            str(self.result_name + "_crf.tif"),
            result,
        )
        self.layer = utils.show_result(
            self._viewer,
            self.result_layer,
            result,
            name="crf_" + self.result_name,
            existing_layer=self.layer,
            colormap="bop orange",
        )

    def _on_start(self):
        self.crf_results = []

    def _on_finish(self):
        self.worker = None
        with contextlib.suppress(RuntimeError):
            self.start_button.setText("Start")

    # should only happen when testing

    def _on_error(self, error):
        logger.error(error)
        self.start_button.setText("Start")
        self.worker.quit()
        self.worker = None
