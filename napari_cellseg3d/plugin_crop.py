import os
import warnings

import napari
import numpy as np
from magicgui import magicgui
from magicgui.widgets import Container
from magicgui.widgets import Slider

# Qt
from qtpy.QtWidgets import QSizePolicy
from tifffile import imwrite

# local
from napari_cellseg3d import interface as ui
from napari_cellseg3d import utils
from napari_cellseg3d.plugin_base import BasePluginSingleImage

DEFAULT_CROP_SIZE = 64


class Cropping(BasePluginSingleImage):
    """A utility plugin for cropping 3D volumes."""

    def __init__(self, viewer: "napari.viewer.Viewer", parent):
        """Creates a Cropping plugin with several buttons :

        * Open file prompt to select volumes directory

        * Open file prompt to select labels directory

        * A dropdown menu with a choice of png or tif filetypes

        * Three spinboxes to choose the dimensions of the cropped volume in x, y, z

        * A button to launch the cropping process (see :doc:`plugin_crop`)

        * A button to close the widget
        """

        super().__init__(viewer, parent)

        self.btn_start = ui.Button("Start", self.start, self)

        self.crop_label_choice = ui.make_checkbox(
            "Crop labels simultaneously", self.toggle_label_path
        )
        self.lbl_label.setVisible(False)
        self.btn_label.setVisible(False)

        self.box_widgets = ui.IntIncrementCounter.make_n(
            3, 1, 1000, DEFAULT_CROP_SIZE
        )
        self.box_lbl = [
            ui.make_label("Size in " + axis + " of cropped volume :", self)
            for axis in "xyz"
        ]

        self.aniso_widgets = ui.AnisotropyWidgets(self)
        ###########
        for box in self.box_widgets:
            box.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._x = 0
        self._y = 0
        self._z = 0
        self._crop_size_x = DEFAULT_CROP_SIZE
        self._crop_size_y = DEFAULT_CROP_SIZE
        self._crop_size_z = DEFAULT_CROP_SIZE

        self.aniso_factors = [1, 1, 1]

        self.image = None
        self.image_layer = None
        self.label = None
        self.label_layer = None

        self.highres_crop_layer = None
        self.labels_crop_layer = None

        self.crop_labels = False

        self.build()

    def toggle_label_path(self):
        if self.crop_label_choice.isChecked():
            self.lbl_label.setVisible(True)
            self.btn_label.setVisible(True)
        else:
            self.lbl_label.setVisible(False)
            self.btn_label.setVisible(False)

    def build(self):
        """Build buttons in a layout and add them to the napari Viewer"""

        w, layout = ui.make_container(0, 0, 1, 11)

        data_group_w, data_group_l = ui.make_group("Data")

        ui.add_widgets(
            data_group_l,
            [
                ui.combine_blocks(self.btn_image, self.lbl_image),
                self.crop_label_choice,  # whether to crop labels or no
                ui.combine_blocks(self.btn_label, self.lbl_label),
                self.file_handling_box,
                self.filetype_choice,
                self.aniso_widgets,
            ],
        )

        self.crop_label_choice.toggle()
        self.toggle_label_path()

        self.filetype_choice.setVisible(False)

        data_group_w.setLayout(data_group_l)
        layout.addWidget(data_group_w)
        ######################
        ui.add_blank(self, layout)
        ######################
        dim_group_w, dim_group_l = ui.make_group("Dimensions")
        [
            dim_group_l.addWidget(widget, alignment=ui.LEFT_AL)
            for list in zip(self.box_lbl, self.box_widgets)
            for widget in list
        ]
        dim_group_w.setLayout(dim_group_l)
        layout.addWidget(dim_group_w)
        #####################
        #####################
        ui.add_blank(self, layout)
        #####################
        #####################
        ui.add_widgets(
            layout,
            [
                self.btn_start,
                self.btn_close,
            ],
        )

        ui.ScrollArea.make_scrollable(layout, self, min_wh=[180, 100])

    def quicksave(self):
        """Quicksaves the cropped volume in the folder from which they originate, with their original file extension.

        * If images are present, saves the cropped version as a single file or image stacks folder depending on what was loaded.

        * If labels are present, saves the cropped version as a single file or 2D stacks folder depending on what was loaded.
        """

        viewer = self._viewer

        time = utils.get_date_time()
        if not self.as_folder:
            if self.image is not None:
                im_filename = os.path.basename(self.image_path).split(".")[0]
                # print(im_filename)
                im_dir = os.path.split(self.image_path)[0] + "/cropped"
                # print(im_dir)
                os.makedirs(im_dir, exist_ok=True)
                viewer.layers["cropped"].save(
                    im_dir + "/" + im_filename + "_cropped_" + time + ".tif"
                )

            # print(self.label)
            if self.label is not None:
                im_filename = os.path.basename(self.label_path).split(".")[0]
                # print(im_filename)
                im_dir = os.path.split(self.label_path)[0] + "/cropped"
                # print(im_dir)
                name = (
                    im_dir
                    + "/"
                    + im_filename
                    + "_labels_cropped_"
                    + time
                    + ".tif"
                )
                dat = viewer.layers["cropped_labels"].data
                os.makedirs(im_dir, exist_ok=True)
                imwrite(name, data=dat)

        else:
            if self.image is not None:

                # im_filename = os.path.basename(self.image_path).split(".")[0]
                im_dir = os.path.split(self.image_path)[0]

                dat = viewer.layers["cropped"].data
                dir_name = im_dir + "/volume_cropped_" + time
                utils.save_stack(dat, dir_name, filetype=self.filetype)

            # print(self.label)
            if self.label is not None:

                # im_filename = os.path.basename(self.image_path).split(".")[0]
                im_dir = os.path.split(self.label_path)[0]

                dir_name = im_dir + "/labels_cropped_" + time
                # print(f"dir name {dir_name}")
                dat = viewer.layers["cropped_labels"].data
                utils.save_stack(dat, dir_name, filetype=self.filetype)

    def check_ready(self):

        if self.image_path == "" or (
            self.crop_labels and self.label_path == ""
        ):
            warnings.warn("Please set all required paths correctly")
            return False
        return True

    def reset(self):
        """Resets all layers and docked widgets"""

        self._viewer.layers.clear()

        self.remove_docked_widgets()

    def start(self):
        """Launches cropping process by loading the files from the chosen folders,
        and adds control widgets to the napari Viewer for moving the cropped volume.
        """

        self.as_folder = self.file_handling_box.isChecked()
        self.filetype = self.filetype_choice.currentText()
        self.crop_labels = self.crop_label_choice.isChecked()

        if self.aniso_widgets.is_enabled():
            self.aniso_factors = (
                self.aniso_widgets.get_anisotropy_resolution_zyx()
            )

        if not self.check_ready():
            return

        self.reset()

        self.image = utils.load_images(
            self.image_path, self.filetype, self.as_folder
        )

        if len(self.image.shape) > 3:
            self.image = np.squeeze(self.image)

        if self.crop_labels:
            self.label = utils.load_images(
                self.label_path, self.filetype, self.as_folder
            )

            if len(self.label.shape) > 3:
                self.label = np.squeeze(
                    self.label
                )  # if channel/batch remnants from MONAI

        vw = self._viewer

        vw.dims.ndisplay = 3
        vw.scale_bar.visible = True

        # add image and labels
        self.image_layer = vw.add_image(
            self.image,
            colormap="inferno",
            contrast_limits=[200, 1000],
            opacity=0.7,
            scale=self.aniso_factors,
        )

        if self.crop_labels:
            self.label_layer = vw.add_labels(
                self.label, scale=self.aniso_factors, visible=False
            )

        @magicgui(call_button="Quicksave")
        def save_widget():
            return self.quicksave()

        save = self._viewer.window.add_dock_widget(
            save_widget, name="", area="left"
        )
        self.docked_widgets.append(save)

        self.add_crop_sliders()

    def add_crop_sliders(
        self,
    ):
        # modified version of code posted by Juan Nunez Iglesias here :
        # https://forum.image.sc/t/napari-viewing-3d-image-of-large-tif-stack-cropping-image-w-general-shape/55500/2
        vw = self._viewer

        image_stack = np.array(self.image)

        self._crop_size_x, self._crop_size_y, self._crop_size_z = [
            box.value() for box in self.box_widgets
        ]

        self._x = 0
        self._y = 0
        self._z = 0

        # print(f"Crop variables")
        # print(image_stack.shape)

        # define crop sizes and boundaries for the image
        crop_sizes = [self._crop_size_x, self._crop_size_y, self._crop_size_z]
        for i in range(len(crop_sizes)):
            if crop_sizes[i] > image_stack.shape[i]:
                crop_sizes[i] = image_stack.shape[i]
                warnings.warn(
                    f"WARNING : Crop dimension in axis {i} was too large at {crop_sizes[i]}, it was set to {image_stack.shape[i]}"
                )
        cropx, cropy, cropz = crop_sizes
        # shapez, shapey, shapex = image_stack.shape
        ends = np.asarray(image_stack.shape) - np.asarray(crop_sizes) + 1

        stepsizes = ends // 100

        # print(crop_sizes)

        # print(ends)
        # print(stepsizes)

        self.highres_crop_layer = vw.add_image(
            image_stack[:cropx, :cropy, :cropz],
            name="cropped",
            blending="additive",
            colormap="twilight_shifted",
            scale=self.image_layer.scale,
        )

        if self.crop_labels:
            label_stack = self.label
            self.labels_crop_layer = vw.add_labels(
                self.label[:cropx, :cropy, :cropz],
                name="cropped_labels",
                scale=self.label_layer.scale,
            )

        def set_slice(
            axis,
            value,
            highres_crop_layer,
            labels_crop_layer=None,
            crop_lbls=False,
        ):
            """ "Update cropped volume position"""
            idx = int(value)
            scale = np.asarray(highres_crop_layer.scale)
            translate = np.asarray(highres_crop_layer.translate)
            izyx = translate // scale
            izyx[axis] = idx
            izyx = [int(var) for var in izyx]
            i, j, k = izyx

            cropx = self._crop_size_x
            cropy = self._crop_size_y
            cropz = self._crop_size_z

            highres_crop_layer.data = image_stack[
                i : i + cropx, j : j + cropy, k : k + cropz
            ]
            highres_crop_layer.translate = scale * izyx
            highres_crop_layer.refresh()

            if crop_lbls and labels_crop_layer is not None:
                labels_crop_layer.data = label_stack[
                    i : i + cropx, j : j + cropy, k : k + cropz
                ]
                labels_crop_layer.translate = scale * izyx
                labels_crop_layer.refresh()

            self._x = i
            self._y = j
            self._z = k

            # spinbox = SpinBox(name="crop_dims", min=1, value=self._crop_size, max=max(image_stack.shape), step=1)
            # spinbox.changed.connect(lambda event : change_size(event))

        sliders = [
            Slider(name=axis, min=0, max=end, step=step)
            for axis, end, step in zip("zyx", ends, stepsizes)
        ]
        for axis, slider in enumerate(sliders):
            slider.changed.connect(
                lambda event, axis=axis: set_slice(
                    axis,
                    event,
                    self.highres_crop_layer,
                    self.labels_crop_layer,
                    self.crop_labels,
                )
            )
        container_widget = Container(layout="vertical")
        container_widget.extend(sliders)
        # vw.window.add_dock_widget([spinbox, container_widget], area="right")
        wdgts = vw.window.add_dock_widget(container_widget, area="right")
        self.docked_widgets.append(wdgts)
        # TEST : trying to dynamically change the size of the cropped volume
        # BROKEN for now
        # @spinbox.changed.connect
        # def change_size(value: int):
        #
        #     print(value)
        #     i = self._x
        #     j = self._y
        #     k = self._z
        #
        #     self._crop_size = value
        #
        #     cropx = value
        #     cropy = value
        #     cropz = value
        #     highres_crop_layer.data = image_stack[
        #         i : i + cropz, j : j + cropy, k : k + cropx
        #     ]
        #     highres_crop_layer.refresh()
        #     labels_crop_layer.data = label_stack[
        #         i : i + cropz, j : j + cropy, k : k + cropx
        #     ]
        #     labels_crop_layer.refresh()
        #


#################################
#################################
#################################
# code for dynamically changing cropped volume with sliders, one for each dim
# WARNING : broken for now

#  def change_size(axis, value) :

#                     print(value)
#                     print(axis)
#                     index = int(value)
#                     scale = np.asarray(highres_crop_layer.scale)
#                     translate = np.asarray(highres_crop_layer.translate)
#                     izyx = translate // scale
#                     izyx[axis] = index
#                     izyx = [int(el) for el in izyx]

#                     cropz,cropy,cropx = izyx

#                     i = self._x
#                     j = self._y
#                     k = self._z

#                     self._crop_size_x = cropx
#                     self._crop_size_y = cropy
#                     self._crop_size_z = cropz


#                     highres_crop_layer.data = image_stack[
#                         i : i + cropz, j : j + cropy, k : k + cropx
#                     ]
#                     highres_crop_layer.refresh()
#                     labels_crop_layer.data = label_stack[
#                         i : i + cropz, j : j + cropy, k : k + cropx
#                     ]
#                     labels_crop_layer.refresh()


#         # @spinbox.changed.connect
#         # spinbox = SpinBox(name=crop_dims, min=1, max=max(image_stack.shape), step=1)
#         # spinbox.changed.connect(lambda event : change_size(event))


#         sliders = [
#             Slider(name=axis, min=0, max=end, step=step)
#             for axis, end, step in zip("zyx", ends, stepsizes)
#         ]
#         for axis, slider in enumerate(sliders):
#             slider.changed.connect(
#                 lambda event, axis=axis: set_slice(axis, event)
#             )

#         spinboxes = [
#             SpinBox(name=axes+" crop size", min=1, value=self._crop_size_init, max=end, step=1)
#             for axes, end in zip("zyx", image_stack.shape)
#         ]
#         for axes, box in enumerate(spinboxes):
#             box.changed.connect(
#                 lambda event, axes=axes : change_size(axis, event)
#             )


#         container_widget = Container(layout="vertical")
#         container_widget.extend(sliders)
#         container_widget.extend(spinboxes)
#         vw.window.add_dock_widget(container_widget, area="right")
