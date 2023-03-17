from dataclasses import dataclass
from typing import List

import numpy as np
import pyclesperanto_prototype as cle
from qtpy.QtWidgets import QWidget
from skimage.filters import thresholding
from skimage.measure import label
from skimage.measure import regionprops
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed
from skimage.transform import resize

# from skimage.measure import mesh_surface_area
# from skimage.measure import marching_cubes
from tifffile import imread

from napari_cellseg3d import interface as ui
from napari_cellseg3d.utils import fill_list_in_between
from napari_cellseg3d.utils import Singleton
from napari_cellseg3d.utils import sphericity_axis

# from napari_cellseg3d.utils import sphericity_volume_area

# list of methods :
WATERSHED = "Watershed"
CONNECTED_COMP = "Connected Components"
VORONOI_OTSU = "Voronoi-Otsu"


class InstanceMethod:
    def __init__(
        self,
        name: str,
        function: callable,
        num_sliders: int,
        num_counters: int,
        widget_parent: QWidget = None,
    ):
        """
        Methods for instance segmentation

        Args:
            name: Name of the instance segmentation method (for UI)
            function: Function to use for instance segmentation
            num_sliders: Number of Slider UI elements needed to set the parameters of the function
            num_counters: Number of DoubleIncrementCounter UI elements needed to set the parameters of the function
            widget_parent: parent for the declared widgets

        """
        self.name = name
        self.function = function
        self.counters: List[ui.DoubleIncrementCounter] = []
        self.sliders: List[ui.Slider] = []
        if num_sliders > 0:
            for i in range(num_sliders):
                widget = f"slider_{i}"
                setattr(
                    self,
                    widget,
                    ui.Slider(
                        0,
                        100,
                        1,
                        divide_factor=100,
                        text_label="",
                        parent=None,
                    ),
                )
                self.sliders.append(getattr(self, widget))

        if num_counters > 0:
            for i in range(num_counters):
                widget = f"counter_{i}"
                setattr(
                    self,
                    widget,
                    ui.DoubleIncrementCounter(label="", parent=None),
                )
                self.counters.append(getattr(self, widget))

    def run_method(self, image):
        raise NotImplementedError("Must be defined in child classes")


class InstanceMethod:
    def __init__(
        self,
        name: str,
        function: callable,
        num_sliders: int,
        num_counters: int,
    ):
        self.name = name
        self.function = function
        self.counters: List[ui.DoubleIncrementCounter] = []
        self.sliders: List[ui.Slider] = []
        if num_sliders > 0:
            for i in range(num_sliders):
                widget = f"slider_{i}"
                setattr(
                    self,
                    widget,
                    ui.Slider(0, 100, 1, divide_factor=100, text_label=""),
                )
                self.sliders.append(getattr(self, widget))

        if num_counters > 0:
            for i in range(num_counters):
                widget = f"counter_{i}"
                setattr(
                    self,
                    widget,
                    ui.DoubleIncrementCounter(label=""),
                )
                self.counters.append(getattr(self, widget))

    def run_method(self, image):
        raise NotImplementedError("Must be defined in child classes")


@dataclass
class ImageStats:
    volume: List[float]
    centroid_x: List[float]
    centroid_y: List[float]
    centroid_z: List[float]
    sphericity_ax: List[float]
    image_size: List[int]
    total_image_volume: int
    total_filled_volume: int
    filling_ratio: float
    number_objects: int

    def get_dict(self):
        return {
            "Volume": self.volume,
            "Centroid x": self.centroid_x,
            "Centroid y": self.centroid_y,
            "Centroid z": self.centroid_z,
            # "Sphericity (volume/area)": sphericity_va,
            "Sphericity (axes)": self.sphericity_ax,
            "Image size": self.image_size,
            "Total image volume": self.total_image_volume,
            "Total object volume (pixels)": self.total_filled_volume,
            "Filling ratio": self.filling_ratio,
            "Number objects": self.number_objects,
        }


def threshold(volume, thresh):
    """Remove all values smaller than the specified threshold in the volume"""
    im = np.squeeze(volume)
    binary = im > thresh
    return np.where(binary, im, np.zeros_like(im))


def voronoi_otsu(
    volume: np.ndarray,
    spot_sigma: float,
    outline_sigma: float,
    remove_small_size: float,
):
    """
    Voronoi-Otsu labeling from pyclesperanto.
    BASED ON CODE FROM : napari_pyclesperanto_assistant by Robert Haase
    https://github.com/clEsperanto/napari_pyclesperanto_assistant

    Args:
        volume (np.ndarray): volume to segment
        spot_sigma (float): parameter determining how close detected objects can be
        outline_sigma (float): determines the smoothness of the segmentation

    Returns:
    Instance segmentation labels from Voronoi-Otsu method

    """
    semantic = np.squeeze(volume)
    instance = cle.voronoi_otsu_labeling(
        semantic, spot_sigma=spot_sigma, outline_sigma=outline_sigma
    )
    # instance = remove_small_objects(instance, remove_small_size)
    return instance


def binary_connected(
    volume: np.array,
    thres=0.5,
    thres_small=3,
):
    r"""Convert binary foreground probability maps to instance masks via
    connected-component labeling.

    Args:
        volume (numpy.ndarray): foreground probability of shape :math:`(C, Z, Y, X)`.
        thres (float): threshold of foreground. Default: 0.8
        thres_small (int): size threshold of small objects to remove. Default: 128
        scale_factors (tuple): scale factors for resizing in :math:`(Z, Y, X)` order. Default: (1.0, 1.0, 1.0)

    """
    logger.debug(
        f"Running connected components segmentation with thres={thres} and thres_small={thres_small}"
    )
    # if len(volume.shape) > 3:
    semantic = np.squeeze(volume)
    foreground = np.where(semantic > thres, volume, 0)  # int(255 * thres)
    segm = label(foreground)
    segm = remove_small_objects(segm, thres_small)

    # if not all(x == 1.0 for x in scale_factors):
    #     target_size = (
    #         int(semantic.shape[0] * scale_factors[0]),
    #         int(semantic.shape[1] * scale_factors[1]),
    #         int(semantic.shape[2] * scale_factors[2]),
    #     )
    #     segm = resize(
    #         segm,
    #         target_size,
    #         order=0,
    #         anti_aliasing=False,
    #         preserve_range=True,
    #     )

    return segm


def binary_watershed(
    volume,
    thres_objects=0.3,
    thres_seeding=0.9,
    thres_small=10,
    rem_seed_thres=3,
):
    r"""Convert binary foreground probability maps to instance masks via
    watershed segmentation algorithm.

    Note:
        This function uses the `skimage.segmentation.watershed <https://github.com/scikit-image/scikit-image/blob/master/skimage/segmentation/_watershed.py#L89>`_
        function that converts the input image into ``np.float64`` data type for processing. Therefore, please make sure enough memory is allocated when handling large arrays.

    Args:
        volume (numpy.ndarray): foreground probability of shape :math:`(C, Z, Y, X)`.
        thres_objects (float): threshold for foreground objects. Default: 0.3
        thres_seeding (float): threshold for seeding. Default: 0.9
        thres_small (int): size threshold of small objects removal. Default: 10
        rem_seed_thres (int): threshold for small seeds removal. Default : 3

    """
    logger.debug(
        f"Running watershed segmentation with thres_objects={thres_objects}, thres_seeding={thres_seeding},"
        f" thres_small={thres_small} and rem_seed_thres={rem_seed_thres}"
    )
    semantic = np.squeeze(volume)
    seed_map = semantic > thres_seeding
    foreground = semantic > thres_objects
    seed = label(seed_map)
    seed = remove_small_objects(seed, rem_seed_thres)
    segm = watershed(-semantic.astype(np.float64), seed, mask=foreground)
    segm = remove_small_objects(segm, thres_small)

    # if not all(x == 1.0 for x in scale_factors):
    #     target_size = (
    #         int(semantic.shape[0] * scale_factors[0]),
    #         int(semantic.shape[1] * scale_factors[1]),
    #         int(semantic.shape[2] * scale_factors[2]),
    #     )
    #     segm = resize(
    #         segm,
    #         target_size,
    #         order=0,
    #         anti_aliasing=False,
    #         preserve_range=True,
    #     )

    return np.array(segm)


def clear_small_objects(image, threshold, is_file_path=False):
    """Calls skimage.remove_small_objects to remove small fragments that might be artifacts.

    Args:
        image: array containing the image
        threshold:  size threshold for removal of objects in pixels. E.g. if 10, all objects smaller than 10 pixels as a whole will be removed.
        is_file_path: if True, will load the image from a file path directly. Default : False

    Returns:
        array: The image with small objects removed
    """

    if is_file_path:
        image = imread(image)

    # print(threshold)

    labeled = label(image)

    result = remove_small_objects(labeled, threshold)

    # print(np.sum(labeled))
    # print(np.sum(result))

    if np.sum(labeled) == np.sum(result):
        print("Warning : no objects were removed")

    if np.amax(image) == 1:
        result = to_semantic(result)

    return result


def to_instance(image, is_file_path=False):
    """Converts a **ground-truth** label to instance (unique id per object) labels. Does not remove small objects.

    Args:
        image: image or path to image
        is_file_path: if True, will consider ``image`` to be a string containing a path to a file, if not treats it as an image data array.

    Returns: resulting converted labels

    """
    if is_file_path:
        image = [imread(image)]
        # image = image.compute()

    result = binary_watershed(
        image, thres_small=0, thres_seeding=0.3, rem_seed_thres=0
    )  # FIXME add params from utils plugin

    return result


def to_semantic(image, is_file_path=False):
    """Converts a **ground-truth** label to semantic (binary 0/1) labels.

    Args:
        image: image or path to image
        is_file_path: if True, will consider ``image`` to be a string containing a path to a file, if not treats it as an image data array.

    Returns: resulting converted labels

    """
    if is_file_path:
        image = imread(image)
        # image = image.compute()

    image[image >= 1] = 1
    result = image.astype(np.uint16)
    return result


def volume_stats(volume_image):
    """Computes various statistics from instance labels and returns them in a dict.
    Currently provided :

        * "Volume": volume of each object
        * "Centroid": x,y,z centroid coordinates for each object
        * "Sphericity (axes)": sphericity computed from semi-minor and semi-major axes
        * "Image size": size of the image
        * "Total image volume": volume in pixels of the whole image
        * "Total object volume (pixels)": total labeled volume in pixels
        * "Filling ratio": ratio of labeled over total pixel volume
        * "Number objects": total number of unique labeled objects

    Args:
        volume_image: instance labels image

    Returns:
        dict: Statistics described above
    """

    properties = regionprops(volume_image)

    # sphericity_va = []
    def sphericity(region):
        try:
            return sphericity_axis(
                region.axis_major_length * 0.5, region.axis_minor_length * 0.5
            )
        except ValueError:
            return (
                np.nan
            )  # FIXME better way ? inconsistent errors in region.axis_minor_length

    sphericity_ax = [sphericity(region) for region in properties]
    # for region in properties:
    # object = (volume_image == region.label).transpose(1, 2, 0)
    # verts, faces, _, values = marching_cubes(
    #     object, level=0, spacing=(1.0, 1.0, 1.0)
    # )
    # surface_area_pixels = mesh_surface_area(verts, faces)
    # sphericity_va.append(
    #     sphericity_volume_area(region.area, surface_area_pixels)
    # )

    volume = [region.area for region in properties]

    def fill(lst, n=len(properties) - 1):
        return fill_list_in_between(lst, n, "")

    if len(volume_image.flatten()) != 0:
        ratio = fill([np.sum(volume) / len(volume_image.flatten())])
    else:
        ratio = 0

    return ImageStats(
        volume,
        [region.centroid[0] for region in properties],
        [region.centroid[0] for region in properties],
        [region.centroid[2] for region in properties],
        sphericity_ax,
        fill([volume_image.shape]),
        fill([len(volume_image.flatten())]),
        fill([np.sum(volume)]),
        ratio,
        fill([len(properties)]),
    )


class Watershed(InstanceMethod, metaclass=Singleton):
    def __init__(self):
        super().__init__(
            name="Watershed",
            function=binary_watershed,
            num_sliders=2,
            num_counters=2,
        )

        self.sliders[0].text_label.setText("Foreground probability threshold")
        self.sliders[
            0
        ].tooltips = "Probability threshold for foreground object"
        self.sliders[0].setValue(50)

        self.sliders[1].text_label.setText("Seed probability threshold")
        self.sliders[1].tooltips = "Probability threshold for seeding"
        self.sliders[1].setValue(90)

        self.counters[0].label.setText("Small object removal")
        self.counters[0].tooltips = (
            "Volume/size threshold for small object removal."
            "\nAll objects with a volume/size below this value will be removed."
        )
        self.counters[0].setValue(30)

        self.counters[1].label.setText("Small seed removal")
        self.counters[1].tooltips = (
            "Volume/size threshold for small seeds removal."
            "\nAll seeds with a volume/size below this value will be removed."
        )
        self.counters[1].setValue(3)

    def run_method(self, image):
        return self.function(
            image,
            self.sliders[0].value(),
            self.sliders[1].value(),
            self.counters[0].value(),
            self.counters[1].value(),
        )


class ConnectedComponents(InstanceMethod, metaclass=Singleton):
    def __init__(self):
        super().__init__(
            name="Connected Components",
            function=binary_connected,
            num_sliders=1,
            num_counters=1,
        )

        self.sliders[0].text_label.setText("Foreground probability threshold")
        self.sliders[
            0
        ].tooltips = "Probability threshold for foreground object"
        self.sliders[0].setValue(80)

        self.counters[0].label.setText("Small objects removal")
        self.counters[0].tooltips = (
            "Volume/size threshold for small object removal."
            "\nAll objects with a volume/size below this value will be removed."
        )
        self.counters[0].setValue(3)

    def run_method(self, image):
        return self.function(
            image, self.sliders[0].value(), self.counters[0].value()
        )


class VoronoiOtsu(InstanceMethod, metaclass=Singleton):
    def __init__(self):
        super().__init__(
            name="Voronoi-Otsu",
            function=voronoi_otsu,
            num_sliders=0,
            num_counters=3,
        )
        self.counters[0].label.setText("Spot sigma")
        self.counters[
            0
        ].tooltips = "Determines how close detected objects can be"
        self.counters[0].setMaximum(100)
        self.counters[0].setValue(2)

        self.counters[1].label.setText("Outline sigma")
        self.counters[
            1
        ].tooltips = "Determines the smoothness of the segmentation"
        self.counters[1].setMaximum(100)
        self.counters[1].setValue(2)

        self.counters[2].label.setText("Small object removal")
        self.counters[2].tooltips = (
            "Volume/size threshold for small object removal."
            "\nAll objects with a volume/size below this value will be removed."
        )

    def run_method(self, image):
        return self.function(
            image,
            self.counters[0].value(),
            self.counters[1].value(),
            self.counters[2].value(),
        )


class InstanceWidgets(QWidget):
    """
    Base widget with several sliders, for use in instance segmentation parameters
    """

    def __init__(self, parent=None):
        """
        Creates an InstanceWidgets widget

        Args:
            parent: parent widget

        """
        super().__init__(parent)

        self.method_choice = ui.DropdownMenu(
            INSTANCE_SEGMENTATION_METHOD_LIST.keys()
        )
        self.methods = {}
        """Contains the instance of the method, with its name as key"""
        self.instance_widgets = {}
        """Contains the lists of widgets for each methods, to show/hide"""

        self.method_choice.currentTextChanged.connect(self._set_visibility)
        self._build()

    def _build(self):

        group = ui.GroupedWidget("Instance segmentation")
        group.layout.addWidget(self.method_choice)

        try:
            for name, method in INSTANCE_SEGMENTATION_METHOD_LIST.items():
                method_class = method(widget_parent=self.parent())
                self.methods[name] = method_class
                self.instance_widgets[name] = []
                # moderately unsafe way to init those widgets
                if len(method_class.sliders) > 0:
                    for slider in method_class.sliders:
                        group.layout.addWidget(slider.container)
                        self.instance_widgets[name].append(slider)
                if len(method_class.counters) > 0:
                    for counter in method_class.counters:
                        group.layout.addWidget(counter.label)
                        group.layout.addWidget(counter)
                        self.instance_widgets[name].append(counter)
        except RuntimeError as e:
            logger.debug(f"Caught runtime error, most likely during testing")

        self.setLayout(group.layout)
        self._set_visibility()

    def _set_visibility(self):
        method = INSTANCE_SEGMENTATION_METHOD_LIST[
            self.method_choice.currentText()
        ]()

        for widget in self.instance_widgets[method.name]:
            widget.set_visibility(True)

        for key in self.instance_widgets.keys():
            if key != method.name:
                for widget in self.instance_widgets[key]:
                    widget.set_visibility(False)

    def run_method(self, volume):
        """
        Calls instance function with chosen parameters

        Args:
            volume: image data to run method on

        Returns: processed image from self._method

        """
        method = INSTANCE_SEGMENTATION_METHOD_LIST[
            self.method_choice.currentText()
        ]()
        return method.run_method(volume)


INSTANCE_SEGMENTATION_METHOD_LIST = {
    Watershed().name: Watershed,
    ConnectedComponents().name: ConnectedComponents,
    VoronoiOtsu().name: VoronoiOtsu,
}
