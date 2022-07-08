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

from napari_cellseg3d import utils
from napari_cellseg3d.plugin_dock import Datamanager


def launch_review(
    original,
    base,
    raw,
    r_path,
    model_type,
    checkbox,
    filetype,
    as_folder,
    zoom_factor,
):
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
    images_original = original
    base_label = base

    viewer = napari.Viewer()

    viewer.scale_bar.visible = True

    viewer.add_image(
        images_original,
        name="volume",
        colormap="inferno",
        contrast_limits=[200, 1000],
        scale=zoom_factor,
    )  # anything bigger than 255 will get mapped to 255... they did it like this because it must have rgb images
    viewer.add_labels(base_label, name="labels", seed=0.6, scale=zoom_factor)

    if raw is not None:  # raw labels is from the prediction
        viewer.add_image(
            ndimage.gaussian_filter(raw, sigma=3),
            colormap="magenta",
            name="low_confident",
            blending="additive",
            scale=zoom_factor,
        )
    else:
        pass

    # def label_and_sort(base_label):  # assigns a different id for every different cell ?
    #     labeled = ndimage.label(base_label, structure=np.ones((3, 3, 3)))[0]
    #
    #     mks, nums = np.unique(labeled, return_counts=True)
    #
    #     idx_list = list(np.argsort(nums[1:]))
    #     nums = np.sort(nums[1:])
    #     labeled_sorted = np.zeros_like(labeled)
    #     for i, idx in enumerate(idx_list):
    #         labeled_sorted = np.where(labeled == mks[1:][idx], i + 1, labeled_sorted)
    #     return labeled_sorted, nums
    #
    # def label_ct(labeled_array, nums, value):
    #     labeled_temp = copy.copy(labeled_array)
    #     idx = np.abs(nums - value).argmin()
    #     labeled_temp = np.where((labeled_temp < idx) & (labeled_temp != 0), 255, 0)
    #     return labeled_temp

    # def show_so_layer(args):
    #     labeled_c, labeled_sorted, nums = args
    #     so_layer = viewer.add_image(labeled_c, colormap='cyan', name='small_object', blending='additive')
    #
    #     object_slider = QSlider(Qt.Horizontal)
    #     object_slider.setMinimum(0)
    #     object_slider.setMaximum(500)
    #     object_slider.setSingleStep(10)
    #     object_slider.setValue(10)
    #
    #     object_slider.valueChanged[int].connect(lambda value=object_slider: calc_object_callback(so_layer, value,
    #                                                                                              labeled_sorted, nums))
    #
    #     lbl = QLabel('object size')
    #
    #     slider_widget = utils.combine_blocks(lbl, object_slider)
    #
    #     viewer.window.add_dock_widget(slider_widget, name='object_size_slider', area='left')
    #
    #     def calc_object_callback(t_layer, value, labeled_array, nums):
    #         t_layer.data = label_ct(labeled_array, nums, value)

    # @thread_worker(connect={"returned": show_so_layer})
    # def create_label():
    #     labeled_sorted, nums = label_and_sort(base_label)
    #     labeled_c = label_ct(labeled_sorted, nums, 10)
    #     return labeled_c, labeled_sorted, nums
    #
    # worker = create_label()
    # if not as_folder:
    #     r_path = os.path.dirname(r_path)

    @magicgui(
        dirname={"mode": "d", "label": "Save labels in... "},
        call_button="Save",
        # call_button_2="Save & quit",
    )
    def file_widget(
        dirname=Path(r_path),
    ):  # file name where to save annotations
        # """Take a filename and do something with it."""
        # print("The filename is:", dirname)

        dirname = Path(r_path)
        # def saver():
        out_dir = file_widget.dirname.value

        # print("The directory is:", out_dir)

        def quicksave():
            if not as_folder:
                if viewer.layers["labels"] is not None:
                    name = os.path.join(str(out_dir), "labels_reviewed.tif")
                    dat = viewer.layers["labels"].data
                    imwrite(name, data=dat)

            else:
                if viewer.layers["labels"] is not None:
                    dir_name = os.path.join(str(out_dir), "labels_reviewed")
                    dat = viewer.layers["labels"].data
                    utils.save_stack(dat, dir_name, filetype=filetype)

        # def quicksave_quit():
        #     quicksave()
        #     viewer.window.close()

        return dirname, quicksave()  # , quicksave_quit()

    # gui = file_widget.show(run=True)  # dirpicker.show(run=True)

    viewer.window.add_dock_widget(file_widget, name=" ", area="bottom")

    # @magicgui(call_button="Save")

    # gui2 = saver.show(run=True)  # saver.show(run=True)
    # viewer.window.add_dock_widget(gui2, name=" ", area="bottom")

    # viewer.window._qt_window.tabifyDockWidget(gui, gui2) #not with FunctionGui ?

    # draw canvas

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
                    zoom_factor,
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
    dmg.prepare(r_path, filetype, model_type, checkbox, as_folder)
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

        if as_folder:
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
