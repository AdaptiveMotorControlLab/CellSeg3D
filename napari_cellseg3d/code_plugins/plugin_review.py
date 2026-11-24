"""Review plugin for 3D labeling of volumes."""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import napari
import numpy as np
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.colors import ListedColormap, Normalize, PowerNorm
from matplotlib.figure import Figure

# Qt
from qtpy.QtCore import QEvent, QTimer
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
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
    """QLineEdit holding the save directory"""
    mouse_callback: Callable | None = None
    dims_callback: Callable | None = None
    paint_event: Any = None
    """labels event that marks the session as modified"""
    layer_events: list[tuple[Any, Callable]] = field(default_factory=list)
    """(layer event, callback) pairs to disconnect when the review ends"""
    hidden: dict[Any, bool] = field(default_factory=dict)
    """layers -> their visibility before the review (review layers included)"""
    locks: dict[Any, Any] = field(default_factory=dict)
    """review layers -> their lock state before the review"""
    selection: list[Any] = field(default_factory=list)
    """selected layers before the review, taken before any layer is added"""
    active: Any = None
    """active layer before the review"""
    modified: bool = False
    """True when labels changed since the last save"""
    datamanager: Any = None
    """Datamanager widget, set once its csv is ready; flushed when the review ends"""


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

        try:
            self.session = self.launch_review()
        except Exception as e:
            logger.exception(f"Could not start the review : {e}")
            self.finish_review(ask=False)
            return
        self.btn_start.setText("Finish review")
        self.session_label.setText(f"Reviewing : {self.session.labels.name}")
        self.session_label.setVisible(True)

    def finish_review(self, ask=True, cancellable=True):
        """Ends the review and restores the viewer as it was before.

        Offers to save first when the labels changed since the last save.

        Args:
            ask (bool): show the save prompt. If False, unsaved changes are kept in the
                layer but not written to disk.
            cancellable (bool): offer Cancel in the save prompt. False when the review
                must end anyway, e.g. its dock was already closed.

        Returns:
            bool: False if the user cancelled, True otherwise.
        """
        session = self.session
        if session is None:
            return True
        if ask and session.modified:
            buttons = QMessageBox.Save | QMessageBox.Discard
            if cancellable:
                buttons |= QMessageBox.Cancel
            choice = QMessageBox.question(
                self,
                "Finish review",
                f"Save the changes to '{session.labels.name}' before finishing ?",
                buttons,
                QMessageBox.Save,
            )
            if choice == QMessageBox.Cancel:
                return False
            if choice == QMessageBox.Save:
                self.save_labels()

        self.session = None  # first, so a second click finds nothing to undo
        self._restore_viewer(session)
        self.btn_start.setText("Start reviewing")
        self.session_label.setVisible(False)
        return True

    def remove_from_viewer(self):
        """Ends the review (asking to save if needed), then closes the plugin."""
        if self.finish_review():
            super().remove_from_viewer()

    def event(self, event):
        """Ends the review when the plugin's dock is closed from its title bar.

        That close only takes the widget out of its dock, without calling
        :py:meth:`remove_from_viewer`. The dock is already gone, so the save prompt
        has no Cancel. Deferred: the dock is still being torn down here.
        """
        if (
            event.type() == QEvent.ParentChange
            and self.parent() is None
            and self.session is not None
        ):
            QTimer.singleShot(0, lambda: self.finish_review(cancellable=False))
        return super().event(event)

    def save_labels(self, directory=None):
        """Writes the reviewed labels to `labels_reviewed.tif`.

        Args:
            directory (str, optional): output folder. Defaults to the one in the save widget.
        """
        session = self.session
        if session is None:
            return None
        if directory is None:
            directory = session.save_dir.text()
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
                # :ndim drops the channel axis of rgb images, as napari does
                np.zeros(volume.data.shape[: volume.ndim], dtype=np.uint16),
                name="labels",
                scale=volume.scale,
                translate=volume.translate,
                rotate=volume.rotate,
                shear=volume.shear,
                affine=volume.affine,
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

        for layer in viewer.layers:
            if layer in review_layers:
                continue
            session.hidden[layer] = layer.visible
            layer.visible = False

        for layer in review_layers:
            session.hidden[layer] = layer.visible
            layer.visible = True
            session.locks[layer] = layer.locked
            layer.locked = True  # blocks deleting the layer from the UI

        viewer.layers.selection.active = session.labels

    def _restore_viewer(self, session):
        """Removes the review widgets and callbacks, and restores the layers.

        Every step runs even if an earlier one fails (a dock already closed, a layer
        removed meanwhile...), so the viewer always ends up back in a normal state.
        """
        viewer = self._viewer

        def attempt(step, action):
            try:
                action()
            except Exception as e:  # keep going: the goal is a usable viewer
                logger.debug(f"Review teardown, {step} : {e}")

        if session.mouse_callback in viewer.mouse_drag_callbacks:
            attempt(
                "mouse callback",
                lambda: viewer.mouse_drag_callbacks.remove(session.mouse_callback),
            )
        if session.dims_callback is not None:
            attempt(
                "slice callback",
                lambda: viewer.dims.events.current_step.disconnect(
                    session.dims_callback
                ),
            )
        if session.paint_event is not None:
            attempt(
                "paint callback",
                lambda: session.paint_event.disconnect(self._mark_modified),
            )
        for emitter, callback in session.layer_events:
            attempt("layer callback", lambda e=emitter, c=callback: e.disconnect(c))

        present = list(viewer.layers)
        for layer, visible in session.hidden.items():
            if layer in present:
                attempt(
                    "visibility", lambda la=layer, v=visible: setattr(la, "visible", v)
                )
        for layer, lock in session.locks.items():
            attempt("lock", lambda la=layer, lo=lock: setattr(la, "locked", lo))

        def restore_selection():
            viewer.layers.selection.clear()
            for layer in session.selection:
                if layer in present:
                    viewer.layers.selection.add(layer)
            if session.active in present:
                viewer.layers.selection.active = session.active

        attempt("selection", restore_selection)

        if session.datamanager is not None:
            # time spent on the current slice is only written on slice changes
            attempt("timer", session.datamanager._update_time_csv)

        for dock in session.docks:
            attempt("dock", lambda d=dock: viewer.window.remove_dock_widget(d))
        session.docks.clear()
        self.docked_widgets = []

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
        selection = list(viewer.layers.selection)
        active = viewer.layers.selection.active
        volume, labels = self._review_layers()
        session = ReviewSession(
            volume=volume, labels=labels, selection=selection, active=active
        )
        # keep around in case of failure below
        self.session = session
        self._focus_on_review(session)
        # napari >= 0.9 moved overlays to viewer.canvas
        canvas = getattr(viewer, "canvas", None)
        scale_bar = (
            canvas.overlays.scale_bar if canvas is not None else viewer.scale_bar
        )
        scale_bar.visible = True

        paint_event = labels.events.paint
        paint_event.connect(self._mark_modified)
        session.paint_event = paint_event

        save_dir = QLineEdit(str(Path(self.config.csv_path)))
        save_dir.setToolTip("Folder where labels_reviewed.tif is written")

        def choose_directory():
            chosen = ui.open_folder_dialog(save_dir, [save_dir.text()])
            if chosen:  # empty when the dialog is cancelled
                save_dir.setText(chosen)

        choose_dir_btn = ui.Button("Choose directory", choose_directory, fixed=False)
        open_dir_btn = ui.Button(
            "Open directory",
            lambda: ui.open_directory(save_dir.text()),
            fixed=False,
        )
        open_dir_btn.setToolTip("Show the save folder in the file browser")
        save_btn = ui.Button("Save", lambda: self.save_labels(), fixed=False)
        # deferred: this button's own dock is removed while finishing
        exit_btn = ui.Button(
            "Exit review",
            lambda: QTimer.singleShot(0, self.finish_review),
            fixed=False,
        )
        exit_btn.setToolTip(
            "Remove the review panels and restore the layers as they were "
            "(asks to save unsaved changes)"
        )
        ui.set_hover_color(exit_btn)

        save_widget = QWidget()
        save_layout = QVBoxLayout(save_widget)
        save_layout.setContentsMargins(0, 0, 0, 0)
        save_layout.addWidget(QLabel("Save labels in..."))
        save_layout.addWidget(save_dir)
        for left, right in ((choose_dir_btn, open_dir_btn), (save_btn, exit_btn)):
            row = QHBoxLayout()
            row.addWidget(left)
            row.addWidget(right)
            save_layout.addLayout(row)

        session.save_dir = save_dir

        # projections side by side, in the bottom bar
        with plt.style.context("dark_background"):
            canvas = FigureCanvas(Figure(figsize=(12, 3.2)))
            canvas.figure.suptitle("Shift-click on image for plot", fontsize=8)

            images = []
            for i, (_, xlabel, ylabel) in enumerate(PROJECTIONS):
                axes = canvas.figure.add_subplot(1, 3, i + 1)
                images.append(
                    axes.imshow(np.zeros((100, 100)), **layer_mpl_style(volume))
                )
                axes.scatter(50, 50, s=30, c="green", alpha=0.6, marker="+")
                axes.set_xlabel(xlabel)
                axes.set_ylabel(ylabel)

            canvas.figure.subplots_adjust(
                left=0.05, bottom=0.18, right=0.98, top=0.88, wspace=0.3
            )

        last_crop = {"volume": None}

        def redraw(event=None):
            """Refills the projections and copies the layer's colormap and contrast."""
            style = layer_mpl_style(volume)
            crop = last_crop["volume"]
            scale = np.asarray(volume.scale[-3:], dtype=float)
            for image, (plane, _, _) in zip(images, PROJECTIONS, strict=True):
                if crop is not None:
                    image.set_data(projection(crop, plane))
                image.set_cmap(style["cmap"])
                image.set_norm(style["norm"])
                # stretch to layer scale
                rows, cols = (axis for axis in range(3) if axis != plane)
                image.axes.set_aspect(scale[rows] / scale[cols])
            canvas.draw_idle()

        for emitter in (
            volume.events.contrast_limits,
            volume.events.gamma,
            volume.events.colormap,
        ):
            emitter.connect(redraw)
            session.layer_events.append((emitter, redraw))

        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        canvas.setMinimumHeight(200)
        session.docks.append(
            viewer.window.add_dock_widget(canvas, name=" ", area="bottom")
        )

        def update_canvas(viewer, event):
            if "shift" not in event.modifiers:
                return
            try:
                # world -> data: drops extra viewer axes, reverse scale/translate
                position = np.asarray(viewer.cursor.position, dtype=float)
                point = np.round(volume.world_to_data(position)[-3:]).astype(int)
                logger.debug(f"plot @ {point}")

                last_crop["volume"] = crop_volume_around_point(point, volume.data)
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
            slice_num=volume_slice_index(viewer, volume),
        )
        session.datamanager = dmg
        # one left dock: slice checked / timer, then save folder, Save and Exit
        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.addWidget(dmg)
        controls_layout.addWidget(save_widget)
        controls_layout.addStretch(1)
        session.docks.append(
            viewer.window.add_dock_widget(controls, name=" ", area="left")
        )

        def update_button(axis_event=None):
            slice_num = volume_slice_index(viewer, volume)
            logger.debug(f"slice num is {slice_num}")
            dmg.update_dm(slice_num)

        viewer.dims.events.current_step.connect(update_button)
        session.dims_callback = update_button
        return session


PROJECTIONS = (
    # (crop axis sliced through the centre, x label, y label) for a (Z, Y, X) crop.
    (0, "X axis", "Y axis"),
    (1, "X axis", "Z axis"),
    (2, "Y axis", "Z axis"),
)


def projection(crop, plane):
    """2D section of a (Z, Y, X) crop through its centre, across the axis `plane`."""
    return np.take(crop, crop.shape[plane] // 2, axis=plane)


def volume_slice_index(viewer, layer):
    """Index along the layer's first data axis (Z) at the viewer's current position.

    The viewer can have more dimensions than the layer, e.g. after a WNet3D inference
    adds a (classes, Z, Y, X) prediction: viewer axis 0 is then the class axis, even
    when that layer is hidden. napari aligns layers to the trailing viewer axes and
    `world_to_data` keeps the trailing coordinates, so leading axes are ignored and the
    layer's own transform (scale, translate...) is applied.
    """
    z = layer.world_to_data(np.asarray(viewer.dims.point, dtype=float))[0]
    return int(np.clip(np.round(z), 0, layer.data.shape[0] - 1))


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


def crop_volume_around_point(points, data):
    """Crops a 100x100x100 volume around a point, padding with zeros at the edges.

    Args:
        points (list): list of 3 integers, the data coordinates of the point to crop around
        data (np.ndarray): the volume to crop
    """
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
