"""Review plugin for 3D labeling of volumes."""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import napari
import numpy as np
from magicgui.widgets import Container, FileEdit, PushButton
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.colors import ListedColormap, Normalize, PowerNorm
from matplotlib.figure import Figure

# Qt
from qtpy.QtWidgets import QLineEdit, QMessageBox, QSizePolicy
from tifffile import imwrite

# local
from napari_cellseg3d import config, utils
from napari_cellseg3d import interface as ui
from napari_cellseg3d.code_plugins.plugin_base import BasePluginSingleImage
from napari_cellseg3d.code_plugins.plugin_review_dock import Datamanager

logger = utils.LOGGER


@dataclass
class ReviewSession:
    """State of a running review, kept so the viewer can be restored when it ends."""

    volume: Any
    """napari.layers.Image being reviewed"""
    labels: Any
    """napari.layers.Labels being edited"""
    docks: list[Any] = field(default_factory=list)
    """dock widgets added for the review"""
    save_dir: Any = None
    """magicgui FileEdit holding the save directory"""
    mouse_callback: Callable | None = None
    dims_callback: Callable | None = None
    paint_event: Any = None
    """labels event that marks the session as modified"""
    layer_events: list[tuple[Any, Callable]] = field(default_factory=list)
    """(layer event, callback) pairs to disconnect when the review ends"""
    hidden: dict[Any, bool] = field(default_factory=dict)
    """other layers -> their visibility before the review"""
    locks: dict[Any, Any] = field(default_factory=dict)
    """review layers -> their lock state before the review"""
    selection: list[Any] = field(default_factory=list)
    active: Any = None
    modified: bool = False
    """True when labels changed since the last save"""


class Reviewer(BasePluginSingleImage, metaclass=ui.QWidgetSingleton):
    """A plugin for selecting volumes and labels file and launching the review process.

    Inherits from : :doc:`plugin_base`.
    """

    def __init__(self, viewer: "napari.viewer.Viewer", parent=None):
        """Creates a Reviewer plugin with several buttons.

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
        self.session: ReviewSession | None = None
        """The running review, or None."""
        self.enable_utils_menu()

        #######################
        # UI
        self.io_panel = self._build_io_panel()

        self.layer_choice.setText("New review")
        self.folder_choice.setText("Existing review")

        self.csv_textbox = QLineEdit(self)
        self.csv_textbox.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.new_csv_choice = ui.CheckBox("Create new dataset ?")

        self.btn_start = ui.Button("Start reviewing", self._on_start_clicked, self)

        self.lbl_mod = ui.make_label("Name", self)

        self.session_label = ui.make_label("", self)
        self.session_label.setVisible(False)

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

        self.image_filewidget.text_field.textChanged.connect(self._update_results_path)
        print(f"{self}")

    def _update_results_path(self):
        p = self.labels_filewidget.text_field.text()
        if p is not None and Path(p).is_file():
            self.results_filewidget.text_field.setText(str(Path(p).parent))

    def _build(self):
        """Build buttons in a layout and add them to the napari Viewer."""
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.MinimumExpanding)

        tab = ui.ContainerWidget(0, 0, 1, 1)
        layout = tab.layout

        # ui.add_blank(self, layout)
        ###########################
        # self.filetype_choice.setVisible(False)
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
            ],
        )

        # self._hide_io_element(self.results_filewidget, self.folder_choice)
        # self._show_io_element(self.results_filewidget)

        self.results_filewidget.text_field.setText(
            str(Path.home() / "cellseg3d" / "review")
        )

        csv_param_w.setLayout(csv_param_l)
        layout.addWidget(csv_param_w)
        ###########################
        ui.add_blank(self, layout)
        ###########################

        ui.add_widgets(
            layout,
            [self.session_label, self.btn_start, self._make_close_button()],
        )

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
        """Checks that images are present and that sizes match."""
        cfg = self.config

        if cfg.image is None:
            raise ValueError("Review requires at least one image")

        if cfg.labels is not None and cfg.image.shape != cfg.labels.shape:
            logger.warning(
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

        self.config.model_name = self.csv_textbox.text()
        self.config.csv_path = self.results_filewidget.text_field.text()

        self.config.new_csv = self.new_csv_choice.isChecked()
        # self.config.filetype = self.filetype

        if self.anisotropy_widgets.enabled():
            zoom = self.anisotropy_widgets.scaling_zyx()
        else:
            zoom = [1, 1, 1]
        self.config.zoom_factor = zoom

    def _on_start_clicked(self):
        if self.session is None:
            self.run_review()
        else:
            self.finish_review()

    def run_review(self):
        """Starts a review session in the current viewer.

        The chosen image and labels layers are reviewed in place (layers are added
        when reviewing from files, and an empty labels layer when none is given).
        Every other layer is hidden, the review layers are protected from deletion,
        and the review widgets are docked. :py:meth:`finish_review` undoes all of it
        and keeps the layers.
        """
        if self.session is not None and not self.finish_review():
            return
        print("New review session\n" + "*" * 20)
        try:
            self._prepare_data()
        except ValueError as e:
            logger.warning(
                f"An exception occurred : {e}. Please ensure you have entered all required parameters."
            )
            return

        self.session = self.launch_review()
        self.btn_start.setText("Finish review")
        self.session_label.setText(f"Reviewing : {self.session.labels.name}")
        self.session_label.setVisible(True)

    def finish_review(self, ask=True):
        """Ends the review and restores the viewer as it was before.

        Offers to save first when the labels changed since the last save.

        Args:
            ask (bool): show the save prompt. If False, unsaved changes are kept in the
                layer but not written to disk.

        Returns:
            bool: False if the user cancelled, True otherwise.
        """
        session = self.session
        if session is None:
            return True
        if ask and session.modified:
            choice = QMessageBox.question(
                self,
                "Finish review",
                f"Save the changes to '{session.labels.name}' before finishing ?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Save,
            )
            if choice == QMessageBox.Cancel:
                return False
            if choice == QMessageBox.Save:
                self.save_labels()

        self._restore_viewer(session)
        self.session = None
        self.btn_start.setText("Start reviewing")
        self.session_label.setVisible(False)
        return True

    def remove_from_viewer(self):
        """Ends the review (asking to save if needed), then closes the plugin."""
        if self.finish_review():
            super().remove_from_viewer()

    def save_labels(self, directory=None):
        """Writes the reviewed labels to `labels_reviewed.tif`.

        Args:
            directory (str, optional): output folder. Defaults to the one in the save widget.
        """
        session = self.session
        if session is None:
            return None
        if directory is None:
            directory = session.save_dir.value
        path = Path(directory) / "labels_reviewed.tif"
        imwrite(str(path), data=session.labels.data)
        session.modified = False
        logger.info(f"Saved labels to {path}")
        return path

    def _mark_modified(self, event=None):
        if self.session is not None:
            self.session.modified = True

    def _review_layers(self):
        """Returns the (image, labels) layers to review, adding them when needed."""
        viewer = self._viewer
        from_layers = self.layer_choice.isChecked()
        if from_layers:
            volume = self.image_layer_loader.layer()
            labels = (
                self.label_layer_loader.layer()
                if self.config.labels is not None
                else None
            )
        else:
            volume = viewer.add_image(
                self.config.image,
                name="volume",
                colormap="inferno",
                contrast_limits=[200, 1000],
            )
            labels = (
                viewer.add_labels(self.config.labels, name="labels")
                if self.config.labels is not None
                else None
            )
        if labels is None:
            labels = viewer.add_labels(
                np.zeros(self.config.image.shape, dtype=np.uint16),
                name="labels",
            )

        # keep the scale of existing layers unless anisotropy was set explicitly
        if not from_layers or self.anisotropy_widgets.enabled():
            volume.scale = self.config.zoom_factor
            labels.scale = self.config.zoom_factor
        return volume, labels

    def _focus_on_review(self, session):
        """Hides other layers, protects the review layers, selects the labels."""
        viewer = self._viewer
        # layers may be napari proxies: compare with ==, `in` and dict keys, never `is`
        review_layers = (session.volume, session.labels)
        session.selection = list(viewer.layers.selection)
        session.active = viewer.layers.selection.active

        for layer in viewer.layers:
            if layer in review_layers:
                continue
            session.hidden[layer] = layer.visible
            layer.visible = False

        for layer in review_layers:
            layer.visible = True
            session.locks[layer] = layer.locked
            layer.locked = True  # blocks deleting the layer from the UI

        viewer.layers.selection.active = session.labels

    def _restore_viewer(self, session):
        """Removes the review widgets and callbacks, and restores the layers."""
        viewer = self._viewer
        if session.mouse_callback in viewer.mouse_drag_callbacks:
            viewer.mouse_drag_callbacks.remove(session.mouse_callback)
        if session.dims_callback is not None:
            viewer.dims.events.current_step.disconnect(session.dims_callback)
        if session.paint_event is not None:
            session.paint_event.disconnect(self._mark_modified)
        for emitter, callback in session.layer_events:
            emitter.disconnect(callback)

        present = list(viewer.layers)
        for layer, visible in session.hidden.items():
            if layer in present:
                layer.visible = visible
        for layer, lock in session.locks.items():
            layer.locked = lock

        viewer.layers.selection.clear()
        for layer in session.selection:
            if layer in present:
                viewer.layers.selection.add(layer)
        if session.active in present:
            viewer.layers.selection.active = session.active

        self.docked_widgets = session.docks
        self.remove_docked_widgets()

    def launch_review(self):
        """Sets up the review in the current viewer and returns its :py:class:`ReviewSession`.

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
        """
        viewer = self._viewer
        volume, labels = self._review_layers()
        session = ReviewSession(volume=volume, labels=labels)
        self._focus_on_review(session)
        viewer.scale_bar.visible = True

        paint_event = labels.events.paint
        paint_event.connect(self._mark_modified)
        session.paint_event = paint_event

        # save folder | Choose directory | Open directory, then Save
        save_dir = FileEdit(
            mode="d", label="Save labels in... ", value=Path(self.config.csv_path)
        )
        open_dir_btn = PushButton(text="Open directory")
        open_dir_btn.tooltip = "Show the save folder in the file browser"
        open_dir_btn.changed.connect(lambda: ui.open_directory(save_dir.value))
        save_btn = PushButton(text="Save")
        save_btn.changed.connect(lambda: self.save_labels())
        save_widget = Container(
            widgets=[
                Container(widgets=[save_dir, open_dir_btn], layout="horizontal"),
                save_btn,
            ]
        )

        session.save_dir = save_dir
        session.docks.append(
            viewer.window.add_dock_widget(save_widget, name=" ", area="bottom")
        )

        with plt.style.context("dark_background"):
            canvas = FigureCanvas(Figure(figsize=(3, 15)))
            canvas.figure.suptitle("Shift-click on image for plot \n", fontsize=8)

            images = []
            for i, (xlabel, ylabel) in enumerate(
                [("X axis", "Y axis"), ("Y axis", "Z axis"), ("X axis", "Z axis")]
            ):
                axes = canvas.figure.add_subplot(3, 1, i + 1)
                images.append(
                    axes.imshow(np.zeros((100, 100)), **layer_mpl_style(volume))
                )
                axes.scatter(50, 50, s=30, c="green", alpha=0.6, marker="+")
                axes.set_xlabel(xlabel)
                axes.set_ylabel(ylabel)

            canvas.figure.subplots_adjust(
                left=0.1, bottom=0.1, right=1, top=0.95, wspace=0, hspace=0.4
            )

        last_crop = {"volume": None}

        def redraw(event=None):
            """Refills the projections and copies the layer's colormap and contrast."""
            style = layer_mpl_style(volume)
            crop = last_crop["volume"]
            for image, plane in zip(images, (0, 1, 2), strict=True):
                if crop is not None:
                    image.set_data(np.moveaxis(crop, plane, 0)[50])
                image.set_cmap(style["cmap"])
                image.set_norm(style["norm"])
            canvas.draw_idle()

        for emitter in (
            volume.events.contrast_limits,
            volume.events.gamma,
            volume.events.colormap,
        ):
            emitter.connect(redraw)
            session.layer_events.append((emitter, redraw))

        canvas.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Maximum)
        session.docks.append(
            viewer.window.add_dock_widget(canvas, name=" ", area="right")
        )

        zoom_factor = self.config.zoom_factor

        def update_canvas(viewer, event):
            if "shift" in event.modifiers:
                try:
                    cursor_position = np.round(viewer.cursor.position).astype(int)
                    logger.debug(f"plot @ {cursor_position}")

                    last_crop["volume"] = crop_volume_around_point(
                        [
                            cursor_position[0],
                            cursor_position[1],
                            cursor_position[2],
                        ],
                        volume.data,
                        zoom_factor,
                    )
                    redraw()
                except Exception as e:
                    logger.exception(e)

        viewer.mouse_drag_callbacks.append(update_canvas)
        session.mouse_callback = update_canvas

        # Qt widget defined in plugin_review_dock.py
        dmg = Datamanager(parent=viewer, image_shape=volume.data.shape)
        dmg.prepare(
            self.config.csv_path,
            self.config.filetype,
            self.config.model_name,
            self.config.new_csv,
        )
        session.docks.append(viewer.window.add_dock_widget(dmg, name=" ", area="left"))

        def update_button(axis_event):
            slice_num = axis_event.value[0]
            logger.debug(f"slice num is {slice_num}")
            dmg.update_dm(slice_num)

        viewer.dims.events.current_step.connect(update_button)
        session.dims_callback = update_button

        for dock in session.docks:
            dock._close_btn = False
        return session


def layer_mpl_style(layer):
    """Matplotlib `cmap` and `norm` matching how napari displays an image layer.

    Uses the layer's colormap, contrast limits and gamma (napari applies gamma as
    ((x - lo) / (hi - lo)) ** gamma, which is what PowerNorm does).
    """
    cmap = ListedColormap(layer.colormap.map(np.linspace(0, 1, 256)))
    lo, hi = layer.contrast_limits
    if layer.gamma == 1:
        norm = Normalize(vmin=lo, vmax=hi, clip=True)
    else:
        norm = PowerNorm(gamma=layer.gamma, vmin=lo, vmax=hi, clip=True)
    return {"cmap": cmap, "norm": norm}


def crop_volume_around_point(points, data, zoom_factor):
    """Crops a 100x100x100 volume around a point, padding with zeros at the edges.

    Args:
        points (list): list of 3 integers, the coordinates of the point to crop around
        data (np.ndarray): the volume to crop
        zoom_factor (list): list of 3 floats, the zoom factor to apply to the volume before cropping
    """
    if list(zoom_factor) != [1, 1, 1]:
        volume = utils.resize(np.asarray(data, dtype=np.float32), zoom_factor)
    else:
        volume = data

    min_coordinates = [point - 50 for point in points]
    max_coordinates = [point + 50 for point in points]
    inferior_bound = [
        min_coordinate if min_coordinate < 0 else 0
        for min_coordinate in min_coordinates
    ]
    superior_bound = [
        (max_coordinate - volume.shape[i] if volume.shape[i] < max_coordinate else 0)
        for i, max_coordinate in enumerate(max_coordinates)
    ]

    crop_slice = tuple(
        slice(np.maximum(0, min_coordinate), max_coordinate)
        for min_coordinate, max_coordinate in zip(
            min_coordinates, max_coordinates, strict=False
        )
    )
    crop_temp = volume[crop_slice]

    cropped_volume = np.zeros((100, 100, 100), dtype=volume.dtype)
    cropped_volume[
        -inferior_bound[0] : 100 - superior_bound[0],
        -inferior_bound[1] : 100 - superior_bound[1],
        -inferior_bound[2] : 100 - superior_bound[2],
    ] = crop_temp
    return cropped_volume
