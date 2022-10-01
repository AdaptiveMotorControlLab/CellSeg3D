from functools import partial
from pathlib import Path
import warnings

import napari
import numpy as np
from magicgui import magicgui
from magicgui.widgets import Container
from magicgui.widgets import Slider

# Qt
from qtpy.QtWidgets import QSizePolicy
from qtpy.QtWidgets import QWidget

# local
from napari_cellseg3d import interface as ui
from napari_cellseg3d.plugin_base import BasePluginSingleImage
from napari_cellseg3d import utils

DEFAULT_CROP_SIZE = 64


class Cropping(BasePluginSingleImage):
    """A utility plugin for cropping 3D volumes."""

    def __init__(self, viewer: "napari.viewer.Viewer", parent=None):
        """Creates a Cropping plugin with several buttons :

        * Open file prompt to select volumes directory

        * Open file prompt to select labels directory

        * A dropdown menu with a choice of png or tif filetypes

        * Three spinboxes to choose the dimensions of the cropped volume in x, y, z

        * A button to launch the cropping process (see :doc:`plugin_crop`)

        * A button to close the widget
        """

        super().__init__(viewer)
        self.docked_widgets = []  # TODO add remove on close
        self.results_path = Path.home() / Path("cellseg3d/cropped")

        self.btn_start = ui.Button("Start", self._start)

        self.image_layer_loader.set_layer_type(napari.layers.Layer)
        self.image_layer_loader.layer_list.label.setText("Image 1")
        # ui.LayerSelecter(self._viewer, "Image 1")
        # self.layer_selection2 = ui.LayerSelecter(self._viewer, "Image 2")
        self.label_layer_loader.set_layer_type(napari.layers.Layer)
        self.label_layer_loader.layer_list.label.setText("Image 2")

        self.crop_second_image_choice = ui.CheckBox(
            "Crop another\nimage simultaneously",
        )
        self.crop_second_image_choice.toggled.connect(
           self._toggle_second_image_io_visibility
        )
        self.crop_second_image_choice.toggled.connect(self._check_image_list)
        self._viewer.layers.events.inserted.connect(self._check_image_list)
        self.folder_choice.clicked.connect(self._toggle_second_image_io_visibility)
        self.layer_choice.clicked.connect(self._toggle_second_image_io_visibility)

        # self.results_filewidget = ui.FilePathWidget(
        #     "Results path",
        #     self._load_results_path,
        #     default=str(self.results_path),
        # )
        # self.results_filewidget.tooltips = str(self.results_path)
        self.results_filewidget.text_field.setText(str(self.results_path))
        self.results_filewidget.check_ready()

        self.crop_size_widgets = ui.IntIncrementCounter.make_n(
            3, 1, 1000, DEFAULT_CROP_SIZE
        )
        self.crop_size_labels = [
            ui.make_label("Size in " + axis + " of cropped volume :", self)
            for axis in "xyz"
        ]

        self.aniso_widgets = ui.AnisotropyWidgets(self)
        ###########
        for box in self.crop_size_widgets:
            box.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._x = 0
        self._y = 0
        self._z = 0
        self._crop_size_x = DEFAULT_CROP_SIZE
        self._crop_size_y = DEFAULT_CROP_SIZE
        self._crop_size_z = DEFAULT_CROP_SIZE

        self.aniso_factors = [1, 1, 1]

        self.image_layer1 = None
        self.image_layer2 = None

        self.im1_crop_layer = None
        self.im2_crop_layer = None

        self.crop_second_image = False

        self._build()
        self._toggle_second_image_io_visibility()

    def _toggle_second_image_io_visibility(self):
        crop_2nd = self.crop_second_image_choice.isChecked()
        if self.layer_choice.isChecked():
            self.label_layer_loader.container().setVisible(crop_2nd)
        elif self.folder_choice.isChecked():
            self.labels_filewidget.setVisible(crop_2nd)

    def _check_image_list(self):

        l1 = self.image_layer_loader.layer_list
        l2 = self.label_layer_loader.layer_list

        if l1.currentText() == l2.currentText():
            try:
                for i in range(l1.count()):
                    if l1.itemText(i) != l2.currentText():
                        l2.setCurrentIndex(i)
            except IndexError:
                return

    def _build(self):
        """Build buttons in a layout and add them to the napari Viewer"""

        container = ui.ContainerWidget(0, 0, 1, 11)
        layout = container.layout

        ui.add_widgets(
            layout,
            [
                self.crop_second_image_choice,
                self._build_io_panel()
            ],
        )
        self.label_layer_loader.container().setVisible(False)
        ######################
        ui.add_blank(self, layout)
        ######################
        dim_group_w, dim_group_l = ui.make_group("Dimensions")

        dim_group_l.addWidget(self.aniso_widgets)
        [
            dim_group_l.addWidget(widget, alignment=ui.ABS_AL)
            for list in zip(self.crop_size_labels, self.crop_size_widgets)
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
            ],
        )

        ui.ScrollArea.make_scrollable(layout, self, min_wh=[200,300])
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.set_io_visibility()

    # def _check_results_path(self, folder):
    #     if folder != "" and isinstance(folder, str):
    #         if not Path(folder).is_dir():
    #             Path(folder).mkdir(parents=True, exist_ok=True)
    #             if not Path(folder).is_dir():
    #                 return False
    #             print(f"Created missing results folder : {folder}")
    #         return True
    #     return False

    # def _load_results_path(self):
    #     """Show file dialog to set :py:attr:`~results_path`"""
    #     folder = ui.open_folder_dialog(self, str(self.results_path))
    #
    #     if self._check_results_path(folder):
    #         self.results_path = Path(folder)
    #         print(f"Results path : {self.results_path}")
            # self.results_filewidget.text_field.setText(str(self.results_path))

    def quicksave(self):
        """Quicksaves the cropped volume in the folder from which they originate, with their original file extension.

        * If images are present, saves the cropped version as a single file or image stacks folder depending on what was loaded.

        * If labels are present, saves the cropped version as a single file or 2D stacks folder depending on what was loaded.
        """

        viewer = self._viewer

        self.check_results_path(str(self.results_path))
        time = utils.get_date_time()

        im1_path = str(
            self.results_path
            / Path("cropped_" + self.image_layer1.name + time)
        )

        viewer.layers[f"cropped_{self.image_layer1.name}"].save(im1_path)

        print(f"Image 1 saved as: {im1_path}")

        if self.crop_second_image:
            im2_path = str(
                self.results_path
                / Path("cropped_" + self.image_layer2.name + time)
            )

            viewer.layers[f"cropped_{self.image_layer2.name}"].save(im2_path)

            print(f"Image 2 saved as: {im2_path}")

    def check_ready(self):

        if self.image_layer_loader.layer_data() is not None:
            if self.crop_second_image:
                if self.label_layer_loader.layer_data() is not None:
                    return True
                else:
                    return False
            return True
        return False

    def _start(self):
        """Launches cropping process by loading the files from the chosen folders,
        and adds control widgets to the napari Viewer for moving the cropped volume.
        """
        if not self.check_ready():
            warnings.warn("Please select at least one valid layer !")
            return

        self._viewer.window.remove_dock_widget(self.parent())

        self.results_path = Path(self.results_filewidget.text_field.text())

        self.crop_second_image = self.crop_second_image_choice.isChecked()

        if self.aniso_widgets.enabled():
            self.aniso_factors = self.aniso_widgets.scaling_zyx()

        self.image_layer1 = self.image_layer_loader.layer()

        if len(self.image_layer1.data) > 3:
            self.image_layer1.data = np.squeeze(self.image_layer1.data)

        if self.crop_second_image:
            self.image_layer2 = self.label_layer_loader.layer()

            if len(self.image_layer2.data.shape) > 3:
                self.image_layer2.data = np.squeeze(
                    self.image_layer2.data
                )  # if channel/batch remnants from MONAI

        vw = self._viewer

        vw.dims.ndisplay = 3
        vw.scale_bar.visible = True

        if self.aniso_widgets.enabled():
            for layer in vw.layers:
                layer.visible = False
                # hide other layers, because of anisotropy

            self.image_layer1 = self.add_isotropic_layer(self.image_layer1)

            if self.crop_second_image:
                self.image_layer2 = self.add_isotropic_layer(
                    self.image_layer2, visible=False
                )
        else:
            self.image_layer1.opacity = 0.7
            self.image_layer1.colormap = "inferno"
            self.image_layer1.contrast_limits = [200, 1000]

            self.image_layer1.refresh()

            if self.crop_second_image:
                self.image_layer2.opacity = 0.7
                self.image_layer2.visible = False

                self.image_layer2.refresh()

        @magicgui(call_button="Quicksave")
        def save_widget():
            return self.quicksave()

        save = self._viewer.window.add_dock_widget(
            save_widget, name="", area="left"
        )
        self.docked_widgets.append(save)

        self._add_crop_sliders()

    def add_isotropic_layer(
        self,
        layer,
        colormap="inferno",
        contrast_lim=[200, 1000],  # TODO generalize ?
        opacity=0.7,
        visible=True,
    ):
        print(layer.name)

        if isinstance(layer, napari.layers.Image):
            layer = self._viewer.add_image(
                layer.data,
                name=f"Scaled_{layer.name}",
                colormap=colormap,
                contrast_limits=contrast_lim,
                opacity=opacity,
                scale=self.aniso_factors,
                visible=visible,
            )
            print("image")
        elif isinstance(layer, napari.layers.Labels):
            layer = self._viewer.add_labels(
                layer.data,
                name=f"Scaled_{layer.name}",
                opacity=opacity,
                scale=self.aniso_factors,
                visible=visible,
            )
            print("label")
        else:
            raise ValueError(
                f"Please select a valid layer type, {type(layer)} is not compatible"
            )
        return layer

    def _add_crop_layer(self, layer, cropx, cropy, cropz):
        if isinstance(layer, napari.layers.Image):
            new_layer = self._viewer.add_image(
                layer.data[:cropx, :cropy, :cropz],
                name=f"cropped_{layer.name}",
                blending="additive",
                colormap="twilight_shifted",
                scale=self.aniso_factors,
            )
        elif isinstance(layer, napari.layers.Labels):
            new_layer = self._viewer.add_labels(
                layer.data[:cropx, :cropy, :cropz],
                name=f"cropped_{layer.name}",
                scale=self.aniso_factors,
            )
        else:
            raise ValueError(
                f"Please select a valid layer type, {type(layer)} is not compatible"
            )
        return new_layer

    def _add_crop_sliders(
        self,
    ):
        # modified version of code posted by Juan Nunez Iglesias here :
        # https://forum.image.sc/t/napari-viewing-3d-image-of-large-tif-stack-cropping-image-w-general-shape/55500/2
        vw = self._viewer

        im1_stack = self.image_layer1.data

        self._crop_size_x, self._crop_size_y, self._crop_size_z = [
            box.value() for box in self.crop_size_widgets
        ]

        self._x = 0
        self._y = 0
        self._z = 0

        # print(f"Crop variables")
        # print(im1_stack.shape)

        # define crop sizes and boundaries for the image
        crop_sizes = [self._crop_size_x, self._crop_size_y, self._crop_size_z]
        for i in range(len(crop_sizes)):
            if crop_sizes[i] > im1_stack.shape[i]:
                crop_sizes[i] = im1_stack.shape[i]
                warnings.warn(
                    f"WARNING : Crop dimension in axis {i} was too large at {crop_sizes[i]}, it was set to {im1_stack.shape[i]}"
                )
        cropx, cropy, cropz = crop_sizes
        ends = np.asarray(im1_stack.shape) - np.asarray(crop_sizes) + 1

        stepsizes = ends // 100

        # print(crop_sizes)
        # print(ends)
        # print(stepsizes)

        self.im1_crop_layer = self._add_crop_layer(
            self.image_layer1, cropx, cropy, cropz
        )

        if self.crop_second_image:
            im2_stack = self.image_layer2.data
            self.im2_crop_layer = self._add_crop_layer(
                self.image_layer2, cropx, cropy, cropz
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

            highres_crop_layer.data = im1_stack[
                i : i + cropx, j : j + cropy, k : k + cropz
            ]
            highres_crop_layer.translate = scale * izyx
            highres_crop_layer.refresh()

            if crop_lbls and labels_crop_layer is not None:
                labels_crop_layer.data = im2_stack[
                    i : i + cropx, j : j + cropy, k : k + cropz
                ]
                labels_crop_layer.translate = scale * izyx
                labels_crop_layer.refresh()

            self._x = i
            self._y = j
            self._z = k

            # spinbox = SpinBox(name="crop_dims", min=1, value=self._crop_size, max=max(im1_stack.shape), step=1)
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
                    self.im1_crop_layer,
                    self.im2_crop_layer,
                    self.crop_second_image,
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
        #     highres_crop_layer.data = im1_stack[
        #         i : i + cropz, j : j + cropy, k : k + cropx
        #     ]
        #     highres_crop_layer.refresh()
        #     labels_crop_layer.data = im2_stack[
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


#                     highres_crop_layer.data = im1_stack[
#                         i : i + cropz, j : j + cropy, k : k + cropx
#                     ]
#                     highres_crop_layer.refresh()
#                     labels_crop_layer.data = im2_stack[
#                         i : i + cropz, j : j + cropy, k : k + cropx
#                     ]
#                     labels_crop_layer.refresh()


#         # @spinbox.changed.connect
#         # spinbox = SpinBox(name=crop_dims, min=1, max=max(im1_stack.shape), step=1)
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
#             for axes, end in zip("zyx", im1_stack.shape)
#         ]
#         for axes, box in enumerate(spinboxes):
#             box.changed.connect(
#                 lambda event, axes=axes : change_size(axis, event)
#             )


#         container_widget = Container(layout="vertical")
#         container_widget.extend(sliders)
#         container_widget.extend(spinboxes)
#         vw.window.add_dock_widget(container_widget, area="right")
