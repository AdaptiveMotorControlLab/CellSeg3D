import os

import napari
import numpy as np
from magicgui import magicgui
from magicgui.widgets import Container
from magicgui.widgets import Slider
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QCheckBox
from qtpy.QtWidgets import QLabel
from qtpy.QtWidgets import QLayout
from qtpy.QtWidgets import QPushButton
from qtpy.QtWidgets import QSizePolicy
from qtpy.QtWidgets import QSpinBox
from qtpy.QtWidgets import QVBoxLayout
from tifffile import imwrite

from napari_cellseg_annotator import utils
from napari_cellseg_annotator.plugin_base import BasePlugin

DEFAULT_CROP_SIZE = 64


class Cropping(BasePlugin):
    """A utility plugin for cropping 3D volumes."""

    def __init__(self, viewer: "napari.viewer.Viewer"):
        """Creates a Cropping plugin with several buttons :

        * Open file prompt to select volumes directory

        * Open file prompt to select labels directory

        * A dropdown menu with a choice of png or tif filetypes

        * Three spinboxes to choose the dimensions of the cropped volume in x, y, z

        * A button to launch the cropping process (see :doc:`plugin_crop`)

        * A button to close the widget
        """

        super().__init__(viewer)

        self.btn_start = QPushButton("Start", self)
        self.btn_start.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn_start.clicked.connect(self.start)

        self.crop_label_choice = QCheckBox("Crop labels as well ?")
        self.crop_label_choice.setSizePolicy(
            QSizePolicy.Fixed, QSizePolicy.Fixed
        )
        self.crop_label_choice.clicked.connect(self.toggle_label_path)
        self.lbl_label.setVisible(False)
        self.btn_label.setVisible(False)

        self.dock_widgets = []  # container of docked widgets for removal

        def make_sizebox_container(ax):

            sizebox = QSpinBox()
            sizebox.setMinimum(1)
            sizebox.setMaximum(1000)
            sizebox.setValue(DEFAULT_CROP_SIZE)
            return sizebox

        self.box_widgets = [make_sizebox_container(ax) for ax in "xyz"]
        self.box_lbl = [
            QLabel("Size in " + axis + " of cropped volume :")
            for axis in "xyz"
        ]

        for box in self.box_widgets:
            box.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._x = 0
        self._y = 0
        self._z = 0
        self._crop_size_x = DEFAULT_CROP_SIZE
        self._crop_size_y = DEFAULT_CROP_SIZE
        self._crop_size_z = DEFAULT_CROP_SIZE

        self.image = None
        self.image_layer = None
        self.label = None
        self.label_layer = None

        #####################################################################
        # TODO remove once done
        self.test_button = True
        if self.test_button:
            self.btntest = QPushButton("test", self)
            self.btntest.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            self.btntest.clicked.connect(self.run_test)
        #####################################################################

        self.build()

    def toggle_label_path(self):
        if self.crop_label_choice:
            self.lbl_label.setVisible(True)
            self.btn_label.setVisible(True)
        else:
            self.lbl_label.setVisible(False)
            self.btn_label.setVisible(False)

    def build(self):
        """Build buttons in a layout and add them to the napari Viewer"""

        vbox = QVBoxLayout()
        vbox.setContentsMargins(0, 0, 1, 11)
        vbox.setSizeConstraint(QLayout.SetFixedSize)

        vbox.addWidget(
            utils.combine_blocks(self.btn_image, self.lbl_image),
            alignment=Qt.AlignmentFlag.AlignLeft,
        )
        vbox.addWidget(
            self.crop_label_choice,
            alignment=Qt.AlignmentFlag.AlignLeft,
        )
        vbox.addWidget(
            utils.combine_blocks(self.btn_label, self.lbl_label),
            alignment=Qt.AlignmentFlag.AlignLeft,
        )

        vbox.addWidget(
            self.file_handling_box, alignment=Qt.AlignmentFlag.AlignLeft
        )
        vbox.addWidget(
            self.filetype_choice, alignment=Qt.AlignmentFlag.AlignLeft
        )
        self.filetype_choice.setVisible(False)
        utils.add_blank(self, vbox)
        [
            vbox.addWidget(widget, alignment=Qt.AlignmentFlag.AlignLeft)
            for list in zip(self.box_widgets, self.box_lbl)
            for widget in list
        ]
        utils.add_blank(self, vbox)
        vbox.addWidget(self.btn_start, alignment=Qt.AlignmentFlag.AlignLeft)
        vbox.addWidget(self.btn_close, alignment=Qt.AlignmentFlag.AlignLeft)

        ##################################################################
        # remove once done ?

        if self.test_button:
            vbox.addWidget(self.btntest, alignment=Qt.AlignmentFlag.AlignLeft)
        ##################################################################

        utils.make_scrollable(
            vbox, self, min_wh=[180, 100], base_wh=[180, 600]
        )
        # self.setLayout(vbox)
        # self.show()
        # self._viewer.window.add_dock_widget(self, name="Crop utility", area="right")

    ###########################################
    # TODO : remove/disable once done
    def run_test(self):

        self.filetype = self.filetype_choice.currentText()
        self.as_folder = self.file_handling_box.isChecked()

        if self.as_folder:
            self.image_path = (
                "C:/Users/Cyril/Desktop/Proj_bachelor/data/visual_png/sample"
            )
            self.label_path = "C:/Users/Cyril/Desktop/Proj_bachelor/data/visual_png/sample_labels"
        else:
            self.image_path = "C:/Users/Cyril/Desktop/Proj_bachelor/data/visual_tif/volumes/images.tif"
            self.label_path = "C:/Users/Cyril/Desktop/Proj_bachelor/data/visual_tif/labels/testing_im.tif"
        self.start()

    ###########################################
    def quicksave(self):

        viewer = self._viewer

        time = utils.get_date_time()
        if not self.as_folder:
            if viewer.layers["cropped"] is not None:
                im_filename = os.path.basename(self.image_path).split(".")[0]
                im_dir = os.path.split(self.image_path)[0] + "/cropped"
                os.makedirs(im_dir, exist_ok=True)
                viewer.layers["cropped"].save(
                    im_dir + "/" + im_filename + "_cropped_" + time + ".tif"
                )

            if viewer.layers["cropped_labels"] is not None:
                im_filename = os.path.basename(self.label_path).split(".")[0]
                im_dir = os.path.split(self.label_path)[0]
                name = (
                    im_dir
                    + "/"
                    + im_filename
                    + "_label_cropped_"
                    + time
                    + ".tif"
                )
                dat = viewer.layers["cropped_labels"].data
                imwrite(name, data=dat)

        else:
            if viewer.layers["cropped"] is not None:
                im_filename = os.path.basename(self.image_path).split(".")[0]
                im_dir = os.path.split(self.image_path)[0]
                dat = viewer.layers["cropped"].data
                dir_name = im_dir + "/cropped_" + time
                utils.save_stack(dat, dir_name, filetype=self.filetype)
            if viewer.layers["cropped_labels"] is not None:

                im_filename = os.path.basename(self.image_path).split(".")[0]
                im_dir = os.path.split(self.label_path)[0]

                dir_name = im_dir + "/label_cropped_" + time
                dat = viewer.layers["cropped_labels"].data
                utils.save_stack(dat, dir_name, filetype=self.filetype)

    def start(self):
        """Launches cropping process by loading the files from the chosen folders,
        and adds control widgets to the napari Viewer for moving the cropped volume.
        """
        self.as_folder = self.file_handling_box.isChecked()
        self.filetype = self.filetype_choice.currentText()
        self.image = utils.load_images(
            self.image_path, self.filetype, self.as_folder
        )
        self.labels = utils.load_images(
            self.label_path, self.filetype, self.as_folder
        )

        vw = self._viewer

        vw.dims.ndisplay = 3

        # add image and labels
        self.image_layer = vw.add_image(
            self.image,
            colormap="inferno",
            contrast_limits=[200, 1000],
            opacity=0.7,
        )
        self.label_layer = vw.add_labels(self.labels, visible=False)

        @magicgui(call_button="Quicksave")
        def save_widget():
            return self.quicksave()

        save = self._viewer.window.add_dock_widget(
            save_widget, name="", area="left"
        )
        self.dock_widgets.append(save)

        self.add_crop_sliders()

    def close(self):
        """Can be re-implemented in children classes"""
        if len(self.dock_widgets) != 0:
            [
                self._viewer.window.remove_dock_widget(w)
                for w in self.dock_widgets
            ]

        self._viewer.window.remove_dock_widget(self)

    def add_crop_sliders(self):

        vw = self._viewer

        label_stack = self.labels
        image_stack = np.array(self.image)

        self._crop_size_x, self._crop_size_y, self._crop_size_z = [
            box.value() for box in self.box_widgets
        ]

        self._x = 0
        self._y = 0
        self._z = 0

        # define crop sizes and boundaries for the image
        crop_sizes = (self._crop_size_x, self._crop_size_y, self._crop_size_z)
        cropx, cropy, cropz = crop_sizes
        # shapez, shapey, shapex = image_stack.shape
        ends = np.asarray(image_stack.shape) - np.asarray(crop_sizes) + 1
        stepsizes = ends // 100

        highres_crop_layer = vw.add_image(
            image_stack[:cropz, :cropy, :cropx],
            name="cropped",
            blending="additive",
            colormap="twilight_shifted",
            scale=self.image_layer.scale,
        )
        labels_crop_layer = vw.add_labels(
            self.labels[:cropz, :cropy, :cropx],
            name="cropped_labels",
            scale=self.label_layer.scale,
        )

        def set_slice(axis, value):
            """ "Update cropped volume posistion"""
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
                i : i + cropz, j : j + cropy, k : k + cropx
            ]
            highres_crop_layer.translate = scale * izyx
            highres_crop_layer.refresh()
            labels_crop_layer.data = label_stack[
                i : i + cropz, j : j + cropy, k : k + cropx
            ]
            labels_crop_layer.translate = scale * izyx
            labels_crop_layer.refresh()
            self._x = k
            self._y = j
            self._z = i

            # spinbox = SpinBox(name="crop_dims", min=1, value=self._crop_size, max=max(image_stack.shape), step=1)
            # spinbox.changed.connect(lambda event : change_size(event))

        sliders = [
            Slider(name=axis, min=0, max=end, step=step)
            for axis, end, step in zip("zyx", ends, stepsizes)
        ]
        for axis, slider in enumerate(sliders):
            slider.changed.connect(
                lambda event, axis=axis: set_slice(axis, event)
            )
        container_widget = Container(layout="vertical")
        container_widget.extend(sliders)
        # vw.window.add_dock_widget([spinbox, container_widget], area="right")
        wdgts = vw.window.add_dock_widget(container_widget, area="right")
        self.dock_widgets.append(wdgts)
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
