import os
from pathlib import Path

import matplotlib.pyplot as plt
import napari
import numpy as np
from magicgui import magicgui
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.figure import Figure
from qtpy.QtWidgets import QSizePolicy
from scipy import ndimage
from tifffile import imwrite

from napari_cellseg3d import config
from napari_cellseg3d import utils
from napari_cellseg3d.plugin_dock import Datamanager


def launch_review(review_config: config.ReviewConfig):
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

    Args:

        original (dask.array.Array): The original images/volumes that have been labeled

        base (dask.array.Array): The labels for the volume

        raw (dask.array.Array): The raw labels from the prediction

        r_path (str): Path to the raw labels

        model_type (str): The name of the model to be displayed in csv filenames.

        checkbox (bool): Whether the "new model" checkbox has been checked or not, to create a new csv

        filetype (str): The file extension of the volumes and labels.

        as_folder (bool): Whether to load as folder or single file

        zoom_factor (array(int)): zoom factors for each axis

    Returns : list of all docked widgets
    """
    images_original = review_config.image
    if review_config.labels is not None:
        base_label = review_config.labels
    else:
        base_label = np.zeros_like(images_original)

    viewer = napari.Viewer()

    viewer.scale_bar.visible = True

    viewer.add_image(
        images_original,
        name="volume",
        colormap="inferno",
        contrast_limits=[200, 1000],
        scale=review_config.zoom_factor,
    )  # anything bigger than 255 will get mapped to 255... they did it like this because it must have rgb images
    viewer.add_labels(
        base_label, name="labels", seed=0.6, scale=review_config.zoom_factor
    )

    @magicgui(
        dirname={"mode": "d", "label": "Save labels in... "},
        call_button="Save",
        # call_button_2="Save & quit",
    )
    def file_widget(
        dirname=Path(review_config.csv_path),
    ):  # file name where to save annotations
        # """Take a filename and do something with it."""
        # print("The filename is:", dirname)

        dirname = Path(review_config.csv_path)
        # def saver():
        out_dir = file_widget.dirname.value

        # print("The directory is:", out_dir)

        def quicksave():
            if not review_config.as_stack:
                if viewer.layers["labels"] is not None:
                    name = os.path.join(str(out_dir), "labels_reviewed.tif")
                    dat = viewer.layers["labels"].data
                    imwrite(name, data=dat)

            else:
                if viewer.layers["labels"] is not None:
                    dir_name = os.path.join(str(out_dir), "labels_reviewed")
                    dat = viewer.layers["labels"].data
                    utils.save_stack(
                        dat, dir_name, filetype=review_config.filetype
                    )

        return dirname, quicksave()

    viewer.window.add_dock_widget(file_widget, name=" ", area="bottom")

    with plt.style.context("dark_background"):
        canvas = FigureCanvas(Figure(figsize=(3, 15)))

        xy_axes = canvas.figure.add_subplot(3, 1, 1)
        canvas.figure.suptitle("Shift-click on image for plot \n", fontsize=8)
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

    viewer.window.add_dock_widget(canvas, name=" ", area="right")

    @viewer.mouse_drag_callbacks.append
    def update_canvas_canvas(viewer, event):

        if "shift" in event.modifiers:
            try:
                cursor_position = np.round(viewer.cursor.position).astype(int)
                print(f"plot @ {cursor_position}")

                cropped_volume = crop_volume_around_point(
                    [
                        cursor_position[0],
                        cursor_position[1],
                        cursor_position[2],
                    ],
                    viewer.layers["volume"],
                    review_config.zoom_factor,
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
                print(e)

    # Qt widget defined in docker.py
    dmg = Datamanager(parent=viewer)
    dmg.prepare(
        review_config.csv_path,
        review_config.filetype,
        review_config.model_name,
        review_config.new_csv,
        review_config.as_stack,
    )
    viewer.window.add_dock_widget(dmg, name=" ", area="left")

    def update_button(axis_event):

        slice_num = axis_event.value[0]
        print(f"slice num is {slice_num}")
        dmg.update(slice_num)

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

        if review_config.as_stack:
            crop_temp = volume[crop_slice].persist().compute()
        else:
            crop_temp = volume[crop_slice]

        cropped_volume = np.zeros((100, 100, 100), np.int16)
        cropped_volume[
            -inferior_bound[0] : 100 - superior_bound[0],
            -inferior_bound[1] : 100 - superior_bound[1],
            -inferior_bound[2] : 100 - superior_bound[2],
        ] = crop_temp
        return cropped_volume

    return viewer, [file_widget, canvas, dmg]
