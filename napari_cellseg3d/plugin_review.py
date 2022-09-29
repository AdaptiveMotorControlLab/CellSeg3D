import warnings
from pathlib import Path

import napari

# Qt
from qtpy.QtWidgets import QLineEdit
from qtpy.QtWidgets import QSizePolicy

# local
from napari_cellseg3d import config
from napari_cellseg3d import interface as ui
from napari_cellseg3d import utils
from napari_cellseg3d.launch_review import launch_review
from napari_cellseg3d.plugin_base import BasePluginSingleImage

warnings.formatwarning = utils.format_Warning


class Reviewer(BasePluginSingleImage):
    """A plugin for selecting volumes and labels file and launching the review process.
    Inherits from : :doc:`plugin_base`"""

    def __init__(self, viewer: "napari.viewer.Viewer", parent=None):
        """Creates a Reviewer plugin with several buttons :

        * Open file prompt to select volumes directory

        * Open file prompt to select labels directory

        * A dropdown menu with a choice of png or tif filetypes

        * A checkbox if you want to create a new status csv for the dataset

        * A button to launch the review process (see :doc:`launch_review`)
        """

        super().__init__(
            viewer,
            parent,
            loads_images=True,
            loads_labels=True,
            has_results=True,
        )

        # self._viewer = viewer # should not be needed
        self.config = config.ReviewConfig()

        #######################
        # UI
        self.io_panel = self._build_io_panel()

        self.layer_choice.setText("New review")
        self.folder_choice.setText("Existing review")

        self.csv_textbox = QLineEdit(self)
        self.csv_textbox.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.new_csv_choice = ui.CheckBox("Create new dataset ?")

        self.btn_start = ui.Button("Start reviewing", self.run_review, self)

        self.lbl_mod = ui.make_label("Name", self)

        self.warn_label = ui.make_label(
            "WARNING : You already have a review session running.\n"
            "Launching another will close the current one,\n"
            " make sure to save your work beforehand",
            None,
        )

        self.anisotropy_widgets = ui.AnisotropyWidgets(
            self, default_x=1.5, default_y=1.5, default_z=5
        )

        ###########################
        # tooltips
        self.csv_textbox.setToolTip("Name of the csv results file")
        self.new_csv_choice.setToolTip(
            "Ignore any pre-existing csv with the specified name and create a new one"
        )
        ###########################

        self.build()

        self.image_filewidget.text_field.textChanged.connect(
            self._update_results_path
        )

    def _update_results_path(self):
        p = self.image_filewidget.text_field.text()
        if p is not None and p != "" and Path(p).is_file():
            self.results_filewidget.text_field.setText(str(Path(p).parent))

    def build(self):
        """Build buttons in a layout and add them to the napari Viewer"""

        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.MinimumExpanding)

        tab = ui.ContainerWidget(0, 0, 1, 1)
        layout = tab.layout

        # ui.add_blank(self, layout)
        ###########################
        self.filetype_choice.setVisible(False)
        layout.addWidget(self.io_panel)
        self.set_io_visibility()
        self.layer_choice.toggle()
        ###########################
        ui.add_blank(self, layout)
        ###########################
        ui.GroupedWidget.create_single_widget_group(
            "Image parameters", self.anisotropy_widgets, layout
        )
        ###########################
        ui.add_blank(self, layout)
        ###########################
        csv_param_w, csv_param_l = ui.make_group("CSV parameters")

        ui.add_widgets(
            csv_param_l,
            [
                ui.combine_blocks(
                    self.csv_textbox,
                    self.lbl_mod,
                    horizontal=False,
                    l=5,
                    t=0,
                    r=5,
                    b=5,
                ),
                self.new_csv_choice,
                self.results_filewidget,
            ],
        )

        # self._hide_io_element(self.results_filewidget, self.folder_choice)
        # self._show_io_element(self.results_filewidget)

        self.results_filewidget.text_field.setText(
            str(Path.home() / Path("cellseg3d_review"))
        )

        csv_param_w.setLayout(csv_param_l)
        layout.addWidget(csv_param_w)
        ###########################
        ui.add_blank(self, layout)
        ###########################

        ui.add_widgets(layout, [self.btn_start, self.make_close_button()])

        ui.ScrollArea.make_scrollable(
            contained_layout=layout, parent=tab, min_wh=[190, 300]
        )

        self.addTab(tab, "Review")

        self.setMinimumSize(180, 100)
        # self.show()
        # self._viewer.window.add_dock_widget(self, name="Reviewer", area="right")
        self.results_filewidget.check_ready()
        self.results_path = self.results_filewidget.text_field.text()

    def check_image_data(self):
        cfg = self.config

        if cfg.image is None:
            raise ValueError("Review requires at least one image")

        if cfg.labels is not None:
            if cfg.image.shape != cfg.labels.shape:
                warnings.warn(
                    "Image and label dimensions do not match ! Please load matching images"
                )

    def prepare_data(self):

        if self.layer_choice.isChecked():
            self.config.image = self.image_layer_loader.layer_data()
            self.config.labels = self.label_layer_loader.layer_data()
        else:
            self.config.image = utils.load_images(
                self.image_filewidget.text_field.text()
            )
            self.config.labels = utils.load_images(
                self.labels_filewidget.text_field.text()
            )

        self.check_image_data()
        self.check_results_path(self.results_filewidget.text_field.text())

        self.config.csv_path = self.results_filewidget.text_field.text()
        self.config.model_name = self.csv_textbox.text()

        self.config.new_csv = self.new_csv_choice.isChecked()
        self.config.filetype = self.filetype_choice.currentText()

        if self.anisotropy_widgets.enabled:
            zoom = self.anisotropy_widgets.scaling_zyx()
        else:
            zoom = [1, 1, 1]
        self.config.zoom_factor = zoom

    def run_review(self):

        """Launches review process by loading the files from the chosen folders,
        and adds several widgets to the napari Viewer.
        If the review process has been launched once before,
        closes the window entirely and launches the review process in a fresh window.

        TODO:

        * Save work done before leaving

        See :doc:`launch_review`

        Returns:
            napari.viewer.Viewer: self.viewer
        """

        print("New review session\n" + "*" * 20)
        previous_viewer = self._viewer
        try:

            self.prepare_data()

            self._viewer, self.docked_widgets = launch_review(
                review_config=self.config
            )
            self.reset()
            previous_viewer.close()
        except ValueError as e:
            warnings.warn(
                f"An exception occurred : {e}. Please ensure you have entered all required parameters."
            )

    def reset(self):
        self.remove_docked_widgets()
