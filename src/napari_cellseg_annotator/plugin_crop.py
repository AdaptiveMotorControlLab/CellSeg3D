import napari
import numpy as np
from magicgui.widgets import Slider, Container
from napari_cellseg_annotator import utils
from napari_cellseg_annotator.plugin_base import BasePlugin
from qtpy.QtWidgets import (
    QVBoxLayout,
    QPushButton,
    QSizePolicy,
    QLabel,
    QSpinBox,
)

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

        def make_sizebox_container(axis: str):
            sizebox = QSpinBox()
            sizebox.setMinimum(1)
            sizebox.setMaximum(1000)
            sizebox.setValue(DEFAULT_CROP_SIZE)
            lblsize = QLabel("Size in " + axis + " of cropped volume :", self)
            return [sizebox, lblsize]

        self.box_widgets = [make_sizebox_container(ax) for ax in "xyz"]

        for wid in self.box_widgets:
            wid[0].setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._x = 0
        self._y = 0
        self._z = 0
        self._crop_size_x = DEFAULT_CROP_SIZE
        self._crop_size_y = DEFAULT_CROP_SIZE
        self._crop_size_z = DEFAULT_CROP_SIZE

        #####################################################################
        # TODO remove once done
        self.test_button = True
        if self.test_button:
            self.btntest = QPushButton("test", self)
            self.btntest.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            self.btntest.clicked.connect(self.run_test)
        #####################################################################

        self.build()

    def build(self):
        """Build buttons in a layout and add them to the napari Viewer"""
        vbox = QVBoxLayout()

        vbox.addWidget(utils.combine_blocks(self.btn_image, self.lbl_image))
        vbox.addWidget(utils.combine_blocks(self.btn_label, self.lbl_label))

        vbox.addWidget(self.file_handling_box)
        self.filetype_choice.setVisible(False)

        [
            vbox.addWidget(utils.combine_blocks(cont[0], cont[1]))
            for cont in self.box_widgets
        ]

        vbox.addWidget(self.btn_start)
        vbox.addWidget(self.btn_close)

        ##################################################################
        # remove once done ?

        if self.test_button:
            vbox.addWidget(self.btntest)
        ##################################################################

        self.setLayout(vbox)
        # self.show()
        # self._viewer.window.add_dock_widget(self, name="Crop utility", area="right")

    ###########################################
    # TODO : remove/disable once done
    def run_test(self):

        self.filetype = self.filetype_choice.currentText()

        if self.file_handling_box.isChecked():
            self.input_path = (
                "C:/Users/Cyril/Desktop/Proj_bachelor/data/visual_png/sample"
            )
            self.label_path = "C:/Users/Cyril/Desktop/Proj_bachelor/data/visual_png/sample_labels"
        else:
            self.input_path = "C:/Users/Cyril/Desktop/Proj_bachelor/data/visual_tif/volumes/images.tif"
            self.label_path = "C:/Users/Cyril/Desktop/Proj_bachelor/data/visual_tif/labels/testing_im.tif"
        self.start()

    ###########################################

    def start(self):
        """Launches cropping process by loading the files from the chosen folders,
        and adds control widgets to the napari Viewer for moving the cropped volume.
        """
        self._crop_size_x, self._crop_size_y, self._crop_size_z = [
            box[0].value() for box in self.box_widgets
        ]
        self.filetype = self.filetype_choice.currentText()
        image = utils.load_images(
            self.input_path, self.filetype, self.file_handling_box.isChecked()
        )
        labels = utils.load_images(
            self.label_path, self.filetype, self.file_handling_box.isChecked()
        )

        vw = self._viewer

        vw.dims.ndisplay = 3

        # add image and labels
        input_image = vw.add_image(
            image, colormap="inferno", contrast_limits=[200, 1000], opacity=0.7
        )
        label_layer = vw.add_labels(labels, visible=False)

        label_stack = labels
        image_stack = np.array(image)

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
            scale=input_image.scale,
        )
        labels_crop_layer = vw.add_labels(
            labels[:cropz, :cropy, :cropx],
            name="cropped_labels",
            scale=label_layer.scale,
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
        # TEST : trying to dynamically change the size of the cropped volume
        # @spinbox.changed.connect
        def change_size(value: int):

            print(value)
            i = self._x
            j = self._y
            k = self._z

            self._crop_size = value

            cropx = value
            cropy = value
            cropz = value
            highres_crop_layer.data = image_stack[
                i : i + cropz, j : j + cropy, k : k + cropx
            ]
            highres_crop_layer.refresh()
            labels_crop_layer.data = label_stack[
                i : i + cropz, j : j + cropy, k : k + cropx
            ]
            labels_crop_layer.refresh()

        container_widget = Container(layout="vertical")
        container_widget.extend(sliders)
        # vw.window.add_dock_widget([spinbox, container_widget], area="right")
        vw.window.add_dock_widget(container_widget, area="right")


#################################
#################################
#################################
# code for mutiple sliders, one for each dim
# broken for now

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
