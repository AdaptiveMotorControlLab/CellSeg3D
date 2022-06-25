from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from magicgui import magicgui
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.figure import Figure
from qtpy.QtWidgets import QSizePolicy
from scipy import ndimage
from monai.transforms import Zoom
from tifffile import imwrite

from napari_cellseg3d import utils
from napari_cellseg3d.plugin_dock import Datamanager


def launch_review(
    viewer,
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
        viewer (napari.viewer.Viewer): The viewer the widgets are to be displayed in

        original (dask.array.Array): The original images/volumes that have been labeled

        base (dask.array.Array): The labels for the volume

        raw (dask.array.Array): The raw labels from the prediction

        r_path (str): Path to the raw labels

        model_type (str): The name of the model to be displayed in csv filenames.

        checkbox (bool): Whether the "new model" checkbox has been checked or not, to create a new csv

        filetype (str): The file extension of the volumes and labels.

        as_folder (bool): Whether to load as folder or single file

        zoom_factor (array(int)): zoom factors for each axis


    """
    images_original = original
    base_label = base

    view1 = viewer

    view1.scale_bar.visible = True

    view1.add_image(
        images_original,
        name="volume",
        colormap="inferno",
        contrast_limits=[200, 1000],
        scale=zoom_factor,
    )  # anything bigger than 255 will get mapped to 255... they did it like this because it must have rgb images
    view1.add_labels(base_label, name="labels", seed=0.6, scale=zoom_factor)

    if raw is not None:  # raw labels is from the prediction
        view1.add_image(
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
    #     so_layer = view1.add_image(labeled_c, colormap='cyan', name='small_object', blending='additive')
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
    #     view1.window.add_dock_widget(slider_widget, name='object_size_slider', area='left')
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

    layer = view1.layers[0]
    layer1 = view1.layers[1]
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
                    time = utils.get_date_time()
                    name = str(out_dir) + "/labels_reviewed_" + time + ".tif"
                    dat = viewer.layers["labels"].data
                    imwrite(name, data=dat)

            else:
                if viewer.layers["labels"] is not None:
                    time = utils.get_date_time()
                    dir_name = str(out_dir) + "/labels_reviewed_" + time
                    dat = viewer.layers["labels"].data
                    utils.save_stack(dat, dir_name, filetype=filetype)

        # def quicksave_quit():
        #     quicksave()
        #     viewer.window.close()

        return dirname, quicksave()  # , quicksave_quit()

    # gui = file_widget.show(run=True)  # dirpicker.show(run=True)

    view1.window.add_dock_widget(file_widget, name=" ", area="bottom")

    # @magicgui(call_button="Save")

    # gui2 = saver.show(run=True)  # saver.show(run=True)
    # view1.window.add_dock_widget(gui2, name=" ", area="bottom")

    # view1.window._qt_window.tabifyDockWidget(gui, gui2) #not with FunctionGui ?

    # draw canvas

    with plt.style.context("dark_background"):
        canvas = FigureCanvas(Figure(figsize=(3, 15)))

        xy_axes = canvas.figure.add_subplot(3, 1, 1)
        canvas.figure.suptitle("Shift-click on image for plot \n", fontsize=8)
        xy_axes.imshow(np.zeros((100, 100), np.int16))
        xy_axes.scatter(50, 50, s=10, c="red", alpha=0.25)
        xy_axes.set_xlabel("x axis")
        xy_axes.set_ylabel("y axis")
        yz_axes = canvas.figure.add_subplot(3, 1, 2)
        yz_axes.imshow(np.zeros((100, 100), np.int16))
        yz_axes.scatter(50, 50, s=10, c="red", alpha=0.25)
        yz_axes.set_xlabel("y axis")
        yz_axes.set_ylabel("z axis")
        zx_axes = canvas.figure.add_subplot(3, 1, 3)
        zx_axes.imshow(np.zeros((100, 100), np.int16))
        zx_axes.scatter(50, 50, s=10, c="red", alpha=0.25)
        zx_axes.set_xlabel("x axis")
        zx_axes.set_ylabel("z axis")

        # canvas.figure.tight_layout()
        canvas.figure.subplots_adjust(
            left=0.1, bottom=0.1, right=1, top=0.95, wspace=0, hspace=0.4
        )

    canvas.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Maximum)

    view1.window.add_dock_widget(canvas, name=" ", area="right")

    @viewer.mouse_drag_callbacks.append
    def update_canvas_canvas(viewer, event):

        if "shift" in event.modifiers:
            try:
                m_point = np.round(viewer.cursor.position).astype(int)
                print(m_point)

                crop_big = crop_img(
                    [m_point[0], m_point[1], m_point[2]],
                    viewer.layers["volume"],
                )

                xy_axes.imshow(crop_big[50], cmap="inferno", vmin=200, vmax=2000)
                yz_axes.imshow(crop_big.transpose(1, 0, 2)[50], cmap="inferno", vmin=200, vmax=2000)
                zx_axes.imshow(crop_big.transpose(2, 0, 1)[50], cmap="inferno", vmin=200, vmax=2000)
                canvas.draw_idle()
            except Exception as e:
                print(e)

    # Qt widget defined in docker.py
    dmg = Datamanager(parent=view1)
    dmg.prepare(r_path, filetype, model_type, checkbox, as_folder)
    view1.window.add_dock_widget(dmg, name=" ", area="left")

    def update_button(axis_event):

        slice_num = axis_event.value[0]
        print(f"slice num is {slice_num}")
        dmg.update(slice_num)

    view1.dims.events.current_step.connect(update_button)

    def crop_img(points, layer):

        if zoom_factor != [1,1,1]:
            im = np.array(layer.data, dtype=np.int16)
            image = Zoom(
                zoom_factor,
                keep_size=False,
                padding_mode="empty",
            )(np.expand_dims(im, axis=0))
            image = image[0]
        # image = ndimage.zoom(layer.data, zoom_factor, mode="nearest") # cleaner but much slower...
        else :
            image = layer.data

        min_vals = [x - 50 for x in points]
        max_vals = [x + 50 for x in points]
        yohaku_minus = [n if n < 0 else 0 for n in min_vals]
        yohaku_plus = [
            x - image.shape[i] if image.shape[i] < x else 0
            for i, x in enumerate(max_vals)
        ]
        crop_slice = tuple(
            slice(np.maximum(0, n), x) for n, x in zip(min_vals, max_vals)
        )

        if as_folder:
            crop_temp = image[crop_slice].persist().compute()
        else:

            crop_temp = layer.data[crop_slice]
        cropped_img = np.zeros((100, 100, 100), np.int16)
        cropped_img[
            -yohaku_minus[0] : 100 - yohaku_plus[0],
            -yohaku_minus[1] : 100 - yohaku_plus[1],
            -yohaku_minus[2] : 100 - yohaku_plus[2],
        ] = crop_temp
        return cropped_img
