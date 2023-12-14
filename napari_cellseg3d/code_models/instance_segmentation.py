"""Instance segmentation methods for 3D images."""
import abc
from dataclasses import dataclass
from functools import partial
from typing import List

import numpy as np
import pyclesperanto_prototype as cle
from qtpy.QtWidgets import QWidget
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed
from tifffile import imread
from tqdm import tqdm

# local
from napari_cellseg3d import interface as ui
from napari_cellseg3d.utils import LOGGER as logger
from napari_cellseg3d.utils import fill_list_in_between, sphericity_axis

# from skimage.measure import marching_cubes
# from skimage.measure import mesh_surface_area
# from napari_cellseg3d.utils import sphericity_volume_area

################
# LIST OF INSTANCE SEGMENTATION METHODS
WATERSHED = "Watershed"
CONNECTED_COMP = "Connected Components"
VORONOI_OTSU = "Voronoi-Otsu"
################

USE_SLIDING_WINDOW = True
"""If True, uses a sliding window to perform instance segmentation to avoid memory issues."""


class InstanceMethod:
    """Base class for instance segmentation methods. Contains the method name, the function to use, and the corresponding UI elements."""

    def __init__(
        self,
        name: str,
        function: callable,
        num_sliders: int,
        num_counters: int,
        widget_parent: QWidget = None,
    ):
        """Methods for instance segmentation.

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
        self._setup_widgets(
            num_counters, num_sliders, widget_parent=widget_parent
        )

        self.recorded_parameters = {}
        """Stores the parameters when calling self.record_parameters()"""

    def _setup_widgets(self, num_counters, num_sliders, widget_parent=None):
        """Initializes the needed widgets for the instance segmentation method, adding sliders and counters to the instance segmentation widget.

        Args:
            num_counters: Number of DoubleIncrementCounter UI elements needed to set the parameters of the function
            num_sliders: Number of Slider UI elements needed to set the parameters of the function
            widget_parent: parent for the declared widgets.
        """
        if num_sliders > 0:
            for i in range(num_sliders):
                widget = f"slider_{i}"
                setattr(
                    self,
                    widget,
                    ui.Slider(
                        lower=0,
                        upper=1000,
                        step=10,
                        divide_factor=1000,
                        text_label="",
                        parent=widget_parent,
                    ),
                )
                self.sliders.append(getattr(self, widget))

        if num_counters > 0:
            for i in range(num_counters):
                widget = f"counter_{i}"
                setattr(
                    self,
                    widget,
                    ui.DoubleIncrementCounter(
                        text_label="", parent=widget_parent
                    ),
                )
                self.counters.append(getattr(self, widget))

    @abc.abstractmethod
    def run_method(self, image):
        """Runs the method on the image with the parameters set in the widget."""
        raise NotImplementedError()

    def _make_list_from_channels(
        self, image
    ):  # TODO(cyril) : adapt to batch dimension (needed ?)
        if len(image.shape) > 4:
            raise ValueError(
                f"Image has {len(image.shape)} dimensions, but should have at most 4 dimensions (CHWD)"
            )
        if len(image.shape) < 2:
            raise ValueError(
                f"Image has {len(image.shape)} dimensions, but should have at least 2 dimensions (HW)"
            )
        if len(image.shape) == 4:
            image = np.squeeze(image)
            if len(image.shape) == 4:
                return [im for im in image]
        return [image]

    def record_parameters(self):
        """Records all the parameters of the instance segmentation method from the current values of the widgets."""
        if len(self.sliders) > 0:
            for slider in self.sliders:
                self.recorded_parameters[
                    slider.label.text()
                ] = slider.slider_value
        if len(self.counters) > 0:
            for counter in self.counters:
                self.recorded_parameters[
                    counter.label.text()
                ] = counter.value()

    def run_method_from_params(self, image):
        """Runs the method on the image with the RECORDED parameters set in the widget.

        See self.record_parameters() and self.run_method()

        Args:
            image: image data to run method on

        Returns: processed image from self._method
        """
        if len(self.recorded_parameters) == 0:
            logger.warning(
                "No parameters recorded, running with values from widgets"
            )
            self.record_parameters()

        parameters = [
            self.recorded_parameters[key] for key in self.recorded_parameters
        ]

        assert len(parameters) == len(self.sliders) + len(self.counters), (
            f"Number of parameters recorded ({len(parameters)}) "
            f"does not match number of sliders ({len(self.sliders)}) "
            f"and counters ({len(self.counters)})"
        )

        return self.function(image, *parameters)

    def run_method_on_channels(self, image):
        """Runs the method on each channel of the image with the parameters set in the widget.

        Args:
            image: image data to run method on

        Returns: processed image from self._method
        """
        image_list = self._make_list_from_channels(image)
        result = np.array([self.run_method(im) for im in image_list])
        return result.squeeze()

    def run_method_on_channels_from_params(self, image):
        """Runs the method on each channel of the image with the RECORDED parameters set in the widget.

        Args:
            image: image data to run method on

        Returns: processed image from self._method
        """
        image_list = self._make_list_from_channels(image)
        result = np.array(
            [self.run_method_from_params(im) for im in image_list]
        )
        return result.squeeze()

    @staticmethod
    def sliding_window(volume, func, patch_size=512, increment_labels=True):
        """Given a volume of dimensions HxWxD, runs the provided function segmentation on the volume using a sliding window of size patch_size.

        If the edge has been reached, the patch size is reduced to fit the remaining space.
        The result is a segmentation of the same size as the input volume.

        Args:
            volume (np.array): The volume to segment
            func (callable): Function to use for instance segmentation. Should be a partial function with the parameters already set.
            patch_size (int): The size of the sliding window.
            increment_labels (bool): If True, increments the labels of each patch by the maximum label of the previous patch.

        Returns:
            np.array: Instance segmentation labels from
        """
        result = np.zeros(volume.shape, dtype=np.uint32)
        max_label_id = 0
        x, y, z = volume.shape[-3:]
        pbar_total = (x // patch_size) * (y // patch_size) * (z // patch_size)
        pbar = tqdm(total=pbar_total)
        for i in range(0, x, patch_size):
            for j in range(0, y, patch_size):
                for k in range(0, z, patch_size):
                    patch = volume[
                        i : min(i + patch_size, x),
                        j : min(j + patch_size, y),
                        k : min(k + patch_size, z),
                    ]
                    patch_result = func(patch)
                    patch_result = np.array(patch_result)
                    # make sure labels are unique, only where result is not 0
                    if increment_labels:
                        patch_result[patch_result > 0] += max_label_id
                        max_label_id = np.max(patch_result)
                    result[
                        i : min(i + patch_size, x),
                        j : min(j + patch_size, y),
                        k : min(k + patch_size, z),
                    ] = patch_result
                    pbar.update(1)
        pbar.close()
        return result


@dataclass
class ImageStats:
    """Dataclass containing various statistics from instance labels."""

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
        """Returns a dict containing the statistics."""
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
    """Remove all values smaller than the specified threshold in the volume."""
    im = np.squeeze(volume)
    binary = im > thresh
    return np.where(binary, im, np.zeros_like(im))


def voronoi_otsu(
    volume: np.ndarray,
    spot_sigma: float,
    outline_sigma: float,
    remove_small_size: float = None,
):
    """Voronoi-Otsu labeling from pyclesperanto.

    BASED ON CODE FROM : napari_pyclesperanto_assistant by Robert Haase
    https://github.com/clEsperanto/napari_pyclesperanto_assistant
    Original code at :
    https://github.com/clEsperanto/pyclesperanto_prototype/blob/master/pyclesperanto_prototype/_tier9/_voronoi_otsu_labeling.py.

    Args:
        volume (np.ndarray): volume to segment
        spot_sigma (float): parameter determining how close detected objects can be
        outline_sigma (float): determines the smoothness of the segmentation
        remove_small_size (float): remove all objects smaller than the specified size in pixel

    Returns:
    Instance segmentation labels from Voronoi-Otsu method

    """
    logger.debug(
        f"Running voronoi otsu segmentation with spot_sigma={spot_sigma} and outline_sigma={outline_sigma}"
    )
    instance = cle.voronoi_otsu_labeling(
        volume, spot_sigma=spot_sigma, outline_sigma=outline_sigma
    )
    instance = np.array(instance)
    if remove_small_size is not None:
        instance = remove_small_objects(instance, remove_small_size)
    return instance


def binary_connected(
    volume: np.array,
    thres=0.5,
    thres_small=3,
):
    r"""Convert binary foreground probability maps to instance masks via connected-component labeling.

    Args:
        volume (numpy.ndarray): foreground probability of shape :math:`(C, Z, Y, X)`.
        thres (float): threshold of foreground. Default: 0.8
        thres_small (int): size threshold of small objects to remove. Default: 128
    """
    logger.debug(
        f"Running connected components segmentation with thres={thres} and thres_small={thres_small}"
    )
    semantic = np.squeeze(volume)
    foreground = np.where(semantic > thres, volume, 0)  # int(255 * thres)
    segm = label(foreground)
    segm = remove_small_objects(segm, thres_small)

    return segm


def binary_watershed(
    volume,
    thres_objects=0.3,
    thres_seeding=0.9,
    thres_small=10,
    rem_seed_thres=3,
):
    r"""Convert binary foreground probability maps to instance masks via watershed segmentation algorithm.

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

    return np.array(segm)


def clear_large_objects(image, large_label_size=200, use_window=True):
    """Uses watershed to label all obejcts, and removes the ones with a volume larger than the specified threshold.

    This is intended for artifact removal, and should not be used for instance segmentation.

    Args:
        image: array containing the image
        large_label_size:  size threshold for removal of objects in pixels. E.g. if 10, all objects larger than 10 pixels as a whole will be removed.
        use_window: if True, will use a sliding window to perform instance segmentation to avoid memory issues. Default : True

    Returns:
        array: The image with large objects removed
    """
    if use_window:
        func = partial(
            binary_watershed,
            thres_objects=0,
            thres_seeding=0,
            thres_small=large_label_size,
            rem_seed_thres=0,
        )
        res = InstanceMethod.sliding_window(
            image, func, increment_labels=False
        )
        return np.where(res > 0, 0, image)

    labeled = binary_watershed(
        image,
        thres_objects=0,
        thres_seeding=0,
        thres_small=large_label_size,
        rem_seed_thres=0,
    )
    return np.where(labeled > 0, 0, image)


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
    return image.astype(np.uint16)


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
    # check if empty or all 0
    if np.sum(volume_image) == 0:
        logger.debug("Skipped empty label image")
        return None

    properties = regionprops(volume_image)

    def sphericity(region):
        try:
            return sphericity_axis(
                region.axis_major_length * 0.5, region.axis_minor_length * 0.5
            )
        except ValueError:
            return (
                np.nan
            )  # FIXME better way ? inconsistent errors in region.axis_minor_length

    sphericities = []
    nan_errors_count = 0
    for region in properties:
        sphericity_ax = sphericity(region)
        if sphericity_axis is None:
            sphericity_ax = np.nan
            nan_errors_count += 1
        sphericities.append(sphericity_ax)
    logger.warning(
        f"{nan_errors_count} invalid sphericities were set to NaN. This occurs for objects with a volume of 1 pixel."
    )
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

    fill = partial(fill_list_in_between, n=len(properties) - 1, fill_value="")

    if len(volume_image.flatten()) != 0:
        ratio = fill([np.sum(volume) / len(volume_image.flatten())])
    else:
        ratio = [0]
        ratio = fill(ratio)

    return ImageStats(
        volume,
        [region.centroid[0] for region in properties],
        [region.centroid[1] for region in properties],
        [region.centroid[2] for region in properties],
        sphericities,
        fill([volume_image.shape]),
        fill([len(volume_image.flatten())]),
        fill([np.sum(volume)]),
        ratio,
        fill([len(properties)]),
    )


class Watershed(InstanceMethod):
    """Widget class for Watershed segmentation. Requires 4 parameters, see binary_watershed."""

    def __init__(self, widget_parent=None):
        """Creates a Watershed widget."""
        super().__init__(
            name=WATERSHED,
            function=binary_watershed,
            num_sliders=2,
            num_counters=2,
            widget_parent=widget_parent,
        )

        self.sliders[0].label.setText("Foreground probability threshold")
        self.sliders[
            0
        ].tooltips = "Probability threshold for foreground object"
        self.sliders[0].setValue(500)

        self.sliders[1].label.setText("Seed probability threshold")
        self.sliders[1].tooltips = "Probability threshold for seeding"
        self.sliders[1].setValue(900)

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

    @property
    def foreground_threshold(self):
        """Returns the value of the foreground threshold slider."""
        return self.sliders[0].slider_value

    @foreground_threshold.setter
    def foreground_threshold(self, value):
        """Sets the value of the foreground threshold slider."""
        self.sliders[0].setValue(value)

    @property
    def seed_threshold(self):
        """Returns the value of the seed threshold slider."""
        return self.sliders[1].slider_value

    @seed_threshold.setter
    def seed_threshold(self, value):
        """Sets the value of the seed threshold slider."""
        self.sliders[1].setValue(value)

    @property
    def small_object_removal(self):
        """Returns the value of the small object removal counter."""
        return self.counters[0].value()

    @small_object_removal.setter
    def small_object_removal(self, value):
        """Sets the value of the small object removal counter."""
        self.counters[0].setValue(value)

    @property
    def small_seed_removal(self):
        """Returns the value of the small seed removal counter."""
        return self.counters[1].value()

    @small_seed_removal.setter
    def small_seed_removal(self, value):
        """Sets the value of the small seed removal counter."""
        self.counters[1].setValue(value)

    def run_method(self, image):
        """Runs the method on the image with the parameters set in the widget.

        If USE_SLIDING_WINDOW is True, uses a sliding window to perform instance segmentation to avoid memory issues.
        """
        if USE_SLIDING_WINDOW:
            func = partial(
                self.function,
                thres_objects=self.sliders[0].slider_value,
                thres_seeding=self.sliders[1].slider_value,
                thres_small=self.counters[0].value(),
                rem_seed_thres=self.counters[1].value(),
            )
            return self.sliding_window(image, func)

        return self.function(
            image,
            self.sliders[0].slider_value,
            self.sliders[1].slider_value,
            self.counters[0].value(),
            self.counters[1].value(),
        )


class ConnectedComponents(InstanceMethod):
    """Widget class for Connected Components instance segmentation. Requires 2 parameters, see binary_connected."""

    def __init__(self, widget_parent=None):
        """Creates a ConnectedComponents widget."""
        super().__init__(
            name=CONNECTED_COMP,
            function=binary_connected,
            num_sliders=1,
            num_counters=1,
            widget_parent=widget_parent,
        )

        self.sliders[0].label.setText("Foreground probability threshold")
        self.sliders[
            0
        ].tooltips = "Probability threshold for foreground object"
        self.sliders[0].setValue(800)

        self.counters[0].label.setText("Small objects removal")
        self.counters[0].tooltips = (
            "Volume/size threshold for small object removal."
            "\nAll objects with a volume/size below this value will be removed."
        )
        self.counters[0].setValue(3)

    @property
    def foreground_threshold(self):
        """Returns the value of the foreground threshold slider."""
        return self.sliders[0].slider_value

    @foreground_threshold.setter
    def foreground_threshold(self, value):
        """Sets the value of the foreground threshold slider."""
        self.sliders[0].setValue(value)

    @property
    def small_object_removal(self):
        """Returns the value of the small object removal counter."""
        return self.counters[0].value()

    @small_object_removal.setter
    def small_object_removal(self, value):
        """Sets the value of the small object removal counter."""
        self.counters[0].setValue(value)

    def run_method(self, image):
        """Runs the method on the image with the parameters set in the widget.

        If USE_SLIDING_WINDOW is True, uses a sliding window to perform instance segmentation to avoid memory issues.
        """
        if USE_SLIDING_WINDOW:
            func = partial(
                self.function,
                thres=self.sliders[0].slider_value,
                thres_small=self.counters[0].value(),
            )
            return self.sliding_window(image, func)

        return self.function(
            image, self.sliders[0].slider_value, self.counters[0].value()
        )


class VoronoiOtsu(InstanceMethod):
    """Widget class for Voronoi-Otsu labeling from pyclesperanto. Requires 2 parameter, see voronoi_otsu."""

    def __init__(self, widget_parent=None):
        """Creates a VoronoiOtsu widget."""
        super().__init__(
            name=VORONOI_OTSU,
            function=voronoi_otsu,
            num_sliders=0,
            num_counters=3,
            widget_parent=widget_parent,
        )
        self.counters[0].label.setText("Spot sigma")  # closeness
        self.counters[
            0
        ].tooltips = "Determines how close detected objects can be"
        self.counters[0].setMaximum(100)
        self.counters[0].setValue(2)

        self.counters[1].label.setText("Outline sigma")  # smoothness
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
        self.counters[2].setValue(1)

    @property
    def spot_sigma(self):
        """Returns the value of the spot sigma counter."""
        return self.counters[0].value()

    @spot_sigma.setter
    def spot_sigma(self, value):
        """Sets the value of the spot sigma counter."""
        self.counters[0].setValue(value)

    @property
    def outline_sigma(self):
        """Returns the value of the outline sigma counter."""
        return self.counters[1].value()

    @outline_sigma.setter
    def outline_sigma(self, value):
        """Sets the value of the outline sigma counter."""
        self.counters[1].setValue(value)

    @property
    def small_object_removal(self):
        """Returns the value of the small object removal counter."""
        return self.counters[2].value()

    @small_object_removal.setter
    def small_object_removal(self, value):
        """Sets the value of the small object removal counter."""
        self.counters[2].setValue(value)

    def run_method(self, image):
        """Runs the method on the image with the parameters set in the widget.

        If USE_SLIDING_WINDOW is True, uses a sliding window to perform instance segmentation to avoid memory issues.
        """
        if USE_SLIDING_WINDOW:
            func = partial(
                self.function,
                spot_sigma=self.counters[0].value(),
                outline_sigma=self.counters[1].value(),
                remove_small_size=self.counters[2].value(),
            )
            return self.sliding_window(image, func)

        return self.function(
            image,
            spot_sigma=self.counters[0].value(),
            outline_sigma=self.counters[1].value(),
            remove_small_size=self.counters[2].value(),
        )


class InstanceWidgets(QWidget):
    """Base widget with several sliders, for use in instance segmentation parameters."""

    def __init__(self, parent=None):
        """Creates an InstanceWidgets widget.

        Args:
            parent: parent widget

        """
        super().__init__(parent)
        self.method_choice = ui.DropdownMenu(
            list(INSTANCE_SEGMENTATION_METHOD_LIST.keys())
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
                # moderately unsafe way to init those widgets ?
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
            logger.debug(
                f"Caught runtime error {e}, most likely during testing"
            )

        self.setLayout(group.layout)
        self._set_visibility()

    def _set_visibility(self):
        for name in self.instance_widgets:
            if name != self.method_choice.currentText():
                for widget in self.instance_widgets[name]:
                    widget.set_visibility(False)
            else:
                for widget in self.instance_widgets[name]:
                    widget.set_visibility(True)

    def run_method(self, volume):
        """Calls instance function with chosen parameters.

        Args:
            volume: image data to run method on

        Returns: processed image from self._method

        """
        method = self.methods[self.method_choice.currentText()]
        return method.run_method_on_channels(volume)


INSTANCE_SEGMENTATION_METHOD_LIST = {
    VORONOI_OTSU: VoronoiOtsu,
    WATERSHED: Watershed,
    CONNECTED_COMP: ConnectedComponents,
}
