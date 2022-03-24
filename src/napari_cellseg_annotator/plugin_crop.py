import numpy as np
from magicgui.widgets import Slider, Container, SpinBox
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QSizePolicy,
    QLabel,
    QComboBox,
)
import napari
from napari_cellseg_annotator import utils


class Cropping(QWidget):
    def __init__(self, viewer: "napari.viewer.Viewer", parent=None):

        super().__init__(parent)

        self._viewer = viewer

        self.input_path = ""
        self.label_path = ""
        self.output_path = ""
        self.filetype = ""

        self.btn1 = QPushButton("Open", self)
        self.btn1.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn1.clicked.connect(self.show_dialog_in)

        self.lbl1 = QLabel("Input directory :")

        self.btn3 = QPushButton("Open", self)
        self.btn3.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn3.clicked.connect(self.show_dialog_lab)

        self.lbl3 = QLabel("Labels directory :", self)

        self.filetype_choice = QComboBox()
        self.filetype_choice.addItems([".png", ".tif"])
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
        self._x = 0
        self._y = 0
        self._z = 0
        self._crop_size = 1
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

        vbox = QVBoxLayout()

        vbox.addWidget(utils.combine_blocks(self.btn1, self.lbl1))
        vbox.addWidget(utils.combine_blocks(self.btn3, self.lbl3))
        vbox.addWidget(self.lblft2)
        vbox.addWidget(utils.combine_blocks(self.filetype_choice, self.lblft))

        vbox.addWidget(self.btn4)
        vbox.addWidget(self.btnc)

        ##################################################################
        # remove once done ?

        if self.test_button:
            vbox.addWidget(self.btntest)
        ##################################################################

        self.setLayout(vbox)
        self.show()

    def show_dialog_in(self):
        default_path = [self.output_path, self.input_path, self.label_path]
        f_name = utils.open_file_dialog(self, default_path)

        if f_name:
            self.input_path = f_name
            self.lbl1.setText(self.input_path)

    def show_dialog_lab(self):
        default_path = [self.output_path, self.input_path, self.label_path]
        f_name = utils.open_file_dialog(self, default_path)

        if f_name:
            self.label_path = f_name
            self.lbl3.setText(self.label_path)

    def close(self):
        """Close the widget"""
        # self.master.setCurrentIndex(0)
        self._viewer.window.remove_dock_widget(self)

    ########################
    # TODO : remove once done
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

    def start(self):
        self.filetype = self.filetype_choice.currentText()

        image = utils.load_images(self.input_path, self.filetype)
        labels = utils.load_images(self.label_path, self.filetype)

        vw = self._viewer

        vw.dims.ndisplay = 3
        input_image = vw.add_image(
            image, colormap="inferno", scale=[1, 1, 1], opacity=0.7
        )
        label_layer = vw.add_labels(labels, visible=False)

        label_stack = labels
        image_stack = np.array(image)

        crop_dims = 64

        crop_sizes = (crop_dims, crop_dims, crop_dims)
        cropz, cropy, cropx = crop_sizes
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

        def set_slice(axis, value, crop_size=64):
            # cropz, cropy, cropx = crop_size
            cropz = int(64)
            cropy = int(64)
            cropx = int(64)
            idx = int(value)
            scale = np.asarray(highres_crop_layer.scale)
            translate = np.asarray(highres_crop_layer.translate)
            izyx = translate // scale
            izyx[axis] = idx
            i, j, k = izyx
            i = int(i)
            j = int(j)
            k = int(k)
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

        def update_crop_size(crop_size):
            self._crop_size = crop_size
            return crop_size

        # spinbox = SpinBox(name="Crop size", min = 1, max = 100, step = 1)
        # spinbox.changed.connect(update_crop_size)

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
        # container_widget.extend(spinbox)
        vw.window.add_dock_widget(container_widget, area="right")
