import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import napari
import numpy as np
from magicgui import magicgui
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.figure import Figure

# Qt
from qtpy.QtWidgets import QLineEdit
from qtpy.QtWidgets import QSizePolicy
from tifffile import imwrite

# local
from napari_cellseg3d import config
from napari_cellseg3d import interface as ui
from napari_cellseg3d import utils
from napari_cellseg3d.code_plugins.plugin_base import BasePluginSingleImage
from napari_cellseg3d.code_plugins.plugin_review_dock import Datamanager

warnings.formatwarning = utils.format_Warning
logger = utils.LOGGER


class Reviewer(BasePluginSingleImage, metaclass=ui.QWidgetSingleton):
    """A plugin for selecting volumes and labels file and launching the review process.
    Inherits from : :doc:`plugin_base`"""

    def __init__(self, viewer: "napari.viewer.Viewer", parent=None):
        """Creates a Reviewer plugin with several buttons :

        * Open file prompt to select volumes directory

        * Open file prompt to select labels directory

        * A dropdown menu with a choice of png or tif filetypes

        * A checkbox if you want to create a new status csv for the dataset

        * A button to launch the review process
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
        self.enable_utils_menu()

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

        self._build()

        self.image_filewidget.text_field.textChanged.connect(
            self._update_results_path
        )
        print(f"{self}")

    def _update_results_path(self):
        p = self.image_filewidget.text_field.text()
        if p is not None and p != "" and Path(p).is_file():
            self.results_filewidget.text_field.setText(str(Path(p).parent))

    def _build(self):
        """Build buttons in a layout and add them to the napari Viewer"""

        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.MinimumExpanding)

        tab = ui.ContainerWidget(0, 0, 1, 1)
        layout = tab.layout

        # ui.add_blank(self, layout)
        ###########################
        self.filetype_choice.setVisible(False)
        layout.addWidget(self.io_panel)
        self._set_io_visibility()
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
            str(
                Path.home() / Path("cellseg3d/review")
            )  # TODO(cyril) : check proper behaviour
        )

        csv_param_w.setLayout(csv_param_l)
        layout.addWidget(csv_param_w)
        ###########################
        ui.add_blank(self, layout)
        ###########################

        ui.add_widgets(layout, [self.btn_start, self._make_close_button()])

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
        """Checks that images are present and that sizes match"""

        cfg = self.config

        if cfg.image is None:
            raise ValueError("Review requires at least one image")

        if cfg.labels is not None:
            if cfg.image.shape != cfg.labels.shape:
                warnings.warn(
                    "Image and label dimensions do not match ! Please load matching images"
                )

    def _prepare_data(self):

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
        self._check_results_path(self.results_filewidget.text_field.text())

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

        See launch_review

        Returns:
            napari.viewer.Viewer: self.viewer
        """

        print("New review session\n" + "*" * 20)
        previous_viewer = self._viewer
        try:

            self._prepare_data()

            self._viewer, self.docked_widgets = self.launch_review()
            self._reset()
            previous_viewer.close()
        except ValueError as e:
            warnings.warn(
                f"An exception occurred : {e}. Please ensure you have entered all required parameters."
            )

    def _reset(self):
        self.remove_docked_widgets()

    def launch_review(self):
        """Launch the review process, loading the original image, the labels & the raw labels (from prediction)
        in the viewer.

        Adds several widgets to the viewer :

        * A data manager widget that lets the user choose a directory to save the labels in, and a save button to quickly
          save.

        * A "checked/not checked" button to let the user confirm that a slice has been checked or not.


              **This writes in a csv file under the corresponding slice the slice status (i.e. checked or not checked)
              to allow tracking of the review process for a given dataset.**

        * A plot widget that, when shift-clicking on the volume or label,
          displays the chosen location on several projections (x-y, y-z, x-z),
          to allow the user to have a better all-around view of the object
          and determine whether it should be labeled or not.

        Returns : list of all docked widgets
        """
        images_original = self.config.image
        if self.config.labels is not None:
            base_label = self.config.labels
        else:
            base_label = np.zeros_like(images_original)

        viewer = napari.Viewer()

        viewer.scale_bar.visible = True

        viewer.add_image(
            images_original,
            name="volume",
            colormap="inferno",
            contrast_limits=[200, 1000],
            scale=self.config.zoom_factor,
        )  # anything bigger than 255 will get mapped to 255... they did it like this because it must have rgb images
        viewer.add_labels(
            base_label, name="labels", seed=0.6, scale=self.config.zoom_factor
        )

        @magicgui(
            dirname={"mode": "d", "label": "Save labels in... "},
            call_button="Save",
            # call_button_2="Save & quit",
        )
        def file_widget(
            dirname=Path(self.config.csv_path),
        ):  # file name where to save annotations
            # """Take a filename and do something with it."""
            # logger.debug(("The filename is:", dirname)

            dirname = Path(self.config.csv_path)
            # def saver():
            out_dir = file_widget.dirname.value

            # logger.debug("The directory is:", out_dir)

            def quicksave():
                # if not self.config.as_stack:
                if viewer.layers["labels"] is not None:
                    name = str(Path(out_dir) / "labels_reviewed.tif")
                    dat = viewer.layers["labels"].data
                    imwrite(name, data=dat)

                # else:
                #     if viewer.layers["labels"] is not None:
                #         dir_name = os.path.join(str(out_dir), "labels_reviewed")
                #         dat = viewer.layers["labels"].data
                #         utils.save_stack(
                #             dat, dir_name, filetype=self.config.filetype
                #         )

            return dirname, quicksave()

        file_widget_dock = viewer.window.add_dock_widget(
            file_widget, name=" ", area="bottom"
        )
        file_widget_dock._close_btn = False

        with plt.style.context("dark_background"):
            canvas = FigureCanvas(Figure(figsize=(3, 15)))

            xy_axes = canvas.figure.add_subplot(3, 1, 1)
            canvas.figure.suptitle(
                "Shift-click on image for plot \n", fontsize=8
            )
            xy_axes.imshow(np.zeros((100, 100), np.int16))
            xy_axes.scatter(50, 50, s=10, c="green", alpha=0.25)
            xy_axes.set_xlabel("x axis")
            xy_axes.set_ylabel("y axis")
            yz_axes = canvas.figure.add_subplot(3, 1, 2)
            yz_axes.imshow(np.zeros((100, 100), np.int16))
            yz_axes.scatter(50, 50, s=10, c="green", alpha=0.25)
            yz_axes.set_xlabel("y axis")
            yz_axes.set_ylabel("z axis")
            zx_axes = canvas.figure.add_subplot(3, 1, 3)
            zx_axes.imshow(np.zeros((100, 100), np.int16))
            zx_axes.scatter(50, 50, s=10, c="green", alpha=0.25)
            zx_axes.set_xlabel("x axis")
            zx_axes.set_ylabel("z axis")

            # canvas.figure.tight_layout()
            canvas.figure.subplots_adjust(
                left=0.1, bottom=0.1, right=1, top=0.95, wspace=0, hspace=0.4
            )

        canvas.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Maximum)

        canvas_dock = viewer.window.add_dock_widget(
            canvas, name=" ", area="right"
        )
        canvas_dock._close_btn = False

        @viewer.mouse_drag_callbacks.append
        def update_canvas_canvas(viewer, event):

            if "shift" in event.modifiers:
                try:
                    cursor_position = np.round(viewer.cursor.position).astype(
                        int
                    )
                    logger.debug(f"plot @ {cursor_position}")

                    cropped_volume = crop_volume_around_point(
                        [
                            cursor_position[0],
                            cursor_position[1],
                            cursor_position[2],
                        ],
                        viewer.layers["volume"],
                        self.config.zoom_factor,
                    )

                    ##########
                    ##########
                    # DEBUG
                    # viewer.add_image(cropped_volume, name="DEBUG_crop_plot")

                    xy_axes.imshow(
                        cropped_volume[50], cmap="inferno", vmin=200, vmax=2000
                    )
                    yz_axes.imshow(
                        cropped_volume.transpose(1, 0, 2)[50],
                        cmap="inferno",
                        vmin=200,
                        vmax=2000,
                    )
                    zx_axes.imshow(
                        cropped_volume.transpose(2, 0, 1)[50],
                        cmap="inferno",
                        vmin=200,
                        vmax=2000,
                    )
                    canvas.draw_idle()
                except Exception as e:
                    logger.error(e)

        # Qt widget defined in docker.py
        dmg = Datamanager(parent=viewer)
        dmg.prepare(
            self.config.csv_path,
            self.config.filetype,
            self.config.model_name,
            self.config.new_csv,
        )
        datamananger = viewer.window.add_dock_widget(
            dmg, name=" ", area="left"
        )
        datamananger._close_btn = False

        def update_button(axis_event):

            slice_num = axis_event.value[0]
            logger.debug(f"slice num is {slice_num}")
            dmg.update_dm(slice_num)

        viewer.dims.events.current_step.connect(update_button)

        def crop_volume_around_point(points, layer, zoom_factor):
            if zoom_factor != [1, 1, 1]:
                data = np.array(layer.data, dtype=np.int16)
                volume = utils.resize(data, zoom_factor)
                # image = ndimage.zoom(layer.data, zoom_factor, mode="nearest") # cleaner but much slower...
            else:
                volume = layer.data

            min_coordinates = [point - 50 for point in points]
            max_coordinates = [point + 50 for point in points]
            inferior_bound = [
                min_coordinate if min_coordinate < 0 else 0
                for min_coordinate in min_coordinates
            ]
            superior_bound = [
                max_coordinate - volume.shape[i]
                if volume.shape[i] < max_coordinate
                else 0
                for i, max_coordinate in enumerate(max_coordinates)
            ]

            crop_slice = tuple(
                slice(np.maximum(0, min_coordinate), max_coordinate)
                for min_coordinate, max_coordinate in zip(
                    min_coordinates, max_coordinates
                )
            )

            # if self.config.as_stack:
            #     crop_temp = volume[crop_slice].persist().compute()
            # else:
            crop_temp = volume[crop_slice]

            cropped_volume = np.zeros((100, 100, 100), np.int16)
            cropped_volume[
                -inferior_bound[0] : 100 - superior_bound[0],
                -inferior_bound[1] : 100 - superior_bound[1],
                -inferior_bound[2] : 100 - superior_bound[2],
            ] = crop_temp
            return cropped_volume

        return viewer, [file_widget, canvas, dmg]
