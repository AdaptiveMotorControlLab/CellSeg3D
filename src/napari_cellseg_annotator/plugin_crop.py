import napari
import numpy as np
from magicgui.widgets import Slider, Container
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QSizePolicy,
    QLabel,
    QComboBox,
    QSpinBox,
)

from napari_cellseg_annotator import utils

DEFAULT_CROP_SIZE = 64


class Cropping(QWidget):
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

        super().__init__(parent)

        self._viewer = viewer
        """napari.viewer.Viewer: viewer in which the widget is displayed"""
        self.input_path = ""
        """str: path to volumes folder"""
        self.label_path = ""
        """str: path to labels folder"""
        # self.output_path = "" # use napari save manually once done

        self.filetype = ""
        """str: filetype, .tif or .png"""

        self._default_path = [self.input_path, self.label_path]

        self.btn1 = QPushButton("Open", self)
        self.btn1.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn1.clicked.connect(self.show_dialog_in)

        self.lbl1 = QLabel("Image directory :")

        self.btn3 = QPushButton("Open", self)
        self.btn3.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn3.clicked.connect(self.show_dialog_lab)

        self.lbl3 = QLabel("Labels directory :", self)

        self.filetype_choice = QComboBox()
        self.filetype_choice.addItems([".tif", ".png"])
        self.filetype_choice.setSizePolicy(
            QSizePolicy.Fixed, QSizePolicy.Fixed
        )

        self.lblft = QLabel("Filetype :", self)
        self.lblft2 = QLabel("(Folders of .png or single .tif files)", self)

        self.btn4 = QPushButton("Start", self)
        self.btn4.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn4.clicked.connect(self.start)

        self.btnc = QPushButton("Close", self)
        self.btnc.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btnc.clicked.connect(self.close)

        ######################
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
        ######################

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

        vbox.addWidget(utils.combine_blocks(self.btn1, self.lbl1))
        vbox.addWidget(utils.combine_blocks(self.btn3, self.lbl3))
        vbox.addWidget(self.lblft2)
        vbox.addWidget(utils.combine_blocks(self.filetype_choice, self.lblft))

        [
            vbox.addWidget(utils.combine_blocks(cont[0], cont[1]))
            for cont in self.box_widgets
        ]

        vbox.addWidget(self.btn4)
        vbox.addWidget(self.btnc)

        ##################################################################
        # remove once done ?

        if self.test_button:
            vbox.addWidget(self.btntest)
        ##################################################################

        self.setLayout(vbox)
        # self.show()
        self._viewer.window.add_dock_widget(self, name="Crop utility", area="right")
    def show_dialog_in(self):
        f_name = utils.open_file_dialog(self, self._default_path)

        if f_name:
            self.input_path = f_name
            self.lbl1.setText(self.input_path)

    def show_dialog_lab(self):
        f_name = utils.open_file_dialog(self, self._default_path)

        if f_name:
            self.label_path = f_name
            self.lbl3.setText(self.label_path)

    def close(self):
        """Close the widget"""
        # self.master.setCurrentIndex(0)
        self._viewer.window.remove_dock_widget(self)

    ###########################################
    # TODO : remove/disable once done
    def run_test(self):

        self.filetype = self.filetype_choice.currentText()

        self.input_path = (
            "C:/Users/Cyril/Desktop/Proj_bachelor/data/visual_png/sample"
        )
        self.label_path = "C:/Users/Cyril/Desktop/Proj_bachelor/data/visual_png/sample_labels"
        if self.filetype == ".tif":
            self.input_path = (
                "C:/Users/Cyril/Desktop/Proj_bachelor/data/visual_tif/volumes"
            )
            self.label_path = (
                "C:/Users/Cyril/Desktop/Proj_bachelor/data/visual_tif/labels"
            )
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
        image = utils.load_images(self.input_path, self.filetype)
        labels = utils.load_images(self.label_path, self.filetype)

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
            colormap="blue",
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
