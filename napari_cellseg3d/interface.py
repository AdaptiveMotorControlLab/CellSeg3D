import warnings
from functools import partial
from typing import List
from typing import Optional

import napari
from qtpy.QtCore import Qt
from qtpy.QtCore import QUrl
from qtpy.QtGui import QDesktopServices
from qtpy.QtWidgets import QCheckBox
from qtpy.QtWidgets import QComboBox
from qtpy.QtWidgets import QDoubleSpinBox
from qtpy.QtWidgets import QFileDialog
from qtpy.QtWidgets import QGridLayout
from qtpy.QtWidgets import QGroupBox
from qtpy.QtWidgets import QHBoxLayout
from qtpy.QtWidgets import QLabel
from qtpy.QtWidgets import QLayout
from qtpy.QtWidgets import QLineEdit
from qtpy.QtWidgets import QPushButton
from qtpy.QtWidgets import QRadioButton
from qtpy.QtWidgets import QScrollArea
from qtpy.QtWidgets import QSizePolicy
from qtpy.QtWidgets import QSlider
from qtpy.QtWidgets import QSpinBox
from qtpy.QtWidgets import QVBoxLayout
from qtpy.QtWidgets import QWidget

from napari_cellseg3d import utils

"""
User interface functions and aliases"""


###############
# aliases
LEFT_AL = Qt.AlignmentFlag.AlignLeft
"""Alias for Qt.AlignmentFlag.AlignLeft, to use in addWidget"""
RIGHT_AL = Qt.AlignmentFlag.AlignRight
"""Alias for Qt.AlignmentFlag.AlignRight, to use in addWidget"""
HCENTER_AL = Qt.AlignmentFlag.AlignHCenter
"""Alias for Qt.AlignmentFlag.AlignHCenter, to use in addWidget"""
CENTER_AL = Qt.AlignmentFlag.AlignCenter
"""Alias for Qt.AlignmentFlag.AlignCenter, to use in addWidget"""
ABS_AL = Qt.AlignmentFlag.AlignAbsolute
"""Alias for Qt.AlignmentFlag.AlignAbsolute, to use in addWidget"""
BOTT_AL = Qt.AlignmentFlag.AlignBottom
"""Alias for Qt.AlignmentFlag.AlignBottom, to use in addWidget"""
###############
# colors
dark_red = "#72071d"  # crimson red
default_cyan = "#8dd3c7"  # turquoise cyan (default matplotlib line color under dark background context)
napari_grey = "#262930"  # napari background color (grey)
napari_param_grey = "#414851"  # napari parameters menu color (lighter gray)
napari_param_darkgrey = "#202228"  # napari default LineEdit color
###############


def toggle_visibility(checkbox, widget):
    """Toggles the visibility of a widget based on the status of a checkbox.

    Args:
        checkbox: The QCheckbox that determines whether to show or not
        widget: The widget to hide or show
    """
    widget.setVisible(checkbox.isChecked())


def add_label(widget, label, label_before=True, horizontal=True):
    if label_before:
        return combine_blocks(widget, label, horizontal=horizontal)
    else:
        return combine_blocks(label, widget, horizontal=horizontal)


def add_label(widget, label, label_before=True, horizontal=True):
    if label_before:
        return combine_blocks(widget, label, horizontal=horizontal)
    else:
        return combine_blocks(label, widget, horizontal=horizontal)


def add_label(widget, label, label_before=True, horizontal=True):
    if label_before:
        return combine_blocks(widget, label, horizontal=horizontal)
    else:
        return combine_blocks(label, widget, horizontal=horizontal)


class RadioButton(QRadioButton):
    def __init__(self, text: str = None, parent=None):

        super().__init__(text, parent)


class Button(QPushButton):
    """Class for a button with a title and connected to a function when clicked. Inherits from QPushButton.

    Args:
        title (str-like): title of the button. Defaults to None, if None no title is set
        func (callable): function to execute when button is clicked. Defaults to None, no binding is made if None
        parent (QWidget): parent QWidget to add button to. Defaults to None, no parent is set if None
        fixed (bool): if True, will set the size policy of the button to Fixed in h and w. Defaults to True.

    """

    def __init__(
        self,
        title: str = None,
        func: callable = None,
        parent: Optional[QWidget] = None,
        fixed: Optional[bool] = True,
    ):
        super().__init__(parent)
        if title is not None:
            self.setText(title)

        if func is not None:
            self.clicked.connect(func)

        if fixed:
            self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    def visibility_condition(self, checkbox):
        """Provide a QCheckBox to use to determine whether to show the button or not"""
        toggle_visibility(checkbox, self)


class DropdownMenu(QComboBox):
    """Creates a dropdown menu with a title and adds specified entries to it"""

    def __init__(
        self,
        entries: Optional[list] = None,
        parent: Optional[QWidget] = None,
        label: Optional[str] = None,
        fixed: Optional[bool] = True,
    ):
        """Args:
        entries (array(str)): Entries to add to the dropdown menu. Defaults to None, no entries if None
        parent (QWidget): parent QWidget to add dropdown menu to. Defaults to None, no parent is set if None
        label (str) : if not None, creates a QLabel with the contents of 'label', and returns the label as well
        fixed (bool): if True, will set the size policy of the dropdown menu to Fixed in h and w. Defaults to True.
        """
        super().__init__(parent)
        self.label = None
        if entries is not None:
            self.addItems(entries)
        if label is not None:
            self.label = QLabel(label)
        if fixed:
            self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)


class CheckBox(QCheckBox):
    """Shortcut class for creating QCheckBox with a title and a function"""

    def __init__(
        self,
        title: Optional[str] = None,
        func: Optional[callable] = None,
        parent: Optional[QWidget] = None,
        fixed: Optional[bool] = True,
    ):
        """
        Args:
            title (str-like): title of the checkbox. Defaults to None, if None no title is set
            func (callable): function to execute when checkbox is toggled. Defaults to None, no binding is made if None
            parent (QWidget): parent QWidget to add checkbox to. Defaults to None, no parent is set if None
            fixed (bool): if True, will set the size policy of the checkbox to Fixed in h and w. Defaults to True.
        """
        super().__init__(title, parent)
        if fixed:
            self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        if func is not None:
            self.toggled.connect(func)


class Slider(QSlider):
    """Shortcut class to create a Slider widget"""

    def __init__(
        self,
        lower: int = 0,
        upper: int = 100,
        step: int = 1,
        default: int = 0,
        divide_factor: float = 1.0,
        parent=None,
        orientation=Qt.Horizontal,
        text_label: str = None,
    ):

        super().__init__(orientation, parent)

        self.setMaximum(upper)
        self.setMinimum(lower)
        self.setSingleStep(step)

        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.text_label = None
        self.container = ContainerWidget(
            # parent=self.parent
        )

        self._divide_factor = divide_factor
        self._value_label = QLineEdit(self.value_text, parent=self)

        if self._divide_factor == 1:
            self._value_label.setFixedWidth(20)
        else:
            self._value_label.setFixedWidth(30)
        self._value_label.setAlignment(Qt.AlignCenter)
        self._value_label.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
        )

        self._value_label.setStyleSheet(
            f"background-color: {napari_param_grey};"
            f"border-radius: 5px;"
            "min - height: 12px;"
            "min - width: 12px;"
        )

        if text_label is not None:
            self.text_label = make_label(text_label, parent=self)

        if default < lower:
            self._warn_outside_bounds(default)
            default = lower
        elif default > upper:
            self._warn_outside_bounds(default)
            default = upper

        self.valueChanged.connect(self._update_value_label)
        self._value_label.textChanged.connect(self._update_slider)

        self.slider_value = default

        self._build_container()

    def _build_container(self):
        self.container.layout

        if self.text_label is not None:
            add_widgets(
                self.container.layout,
                [
                    self.text_label,
                    combine_blocks(self._value_label, self, b=0),
                ],
            )
        else:
            add_widgets(
                self.container.layout,
                [combine_blocks(self._value_label, self, b=0)],
            )

    def _warn_outside_bounds(self, default):
        warnings.warn(
            f"Default value {default} was outside of the ({self.minimum()}:{self.maximum()}) range"
        )

    def _update_slider(self):
        """Update slider when value is changed"""
        if self._value_label.text() == "":
            return

        value = float(self._value_label.text()) * self._divide_factor

        if value < self.minimum():
            self.slider_value = self.minimum()
            return
        if value > self.maximum():
            self.slider_value = self.maximum()
            return

        self.slider_value = value

    def _update_value_label(self):
        """Update label, to connect to when slider is dragged"""
        self._value_label.setText(str(self.value_text))

    @property
    def tooltips(self):
        return self.toolTip()

    @tooltips.setter
    def tooltips(self, tooltip: str):
        self.setToolTip(tooltip)
        self._value_label.setToolTip(tooltip)

        if self.text_label is not None:
            self.text_label.setToolTip(tooltip)

    @property
    def slider_value(self):
        """Get value of the slider divided by self._divide_factor to implement floats in Slider"""
        if self._divide_factor == 1.0:
            return self.value()

        try:
            return self.value() / self._divide_factor
        except ZeroDivisionError as e:
            raise ZeroDivisionError(
                f"Divide factor cannot be 0 for Slider : {e}"
            )

    @property
    def value_text(self):
        """Get value of the slide bar as string"""
        return str(self.slider_value)

    @slider_value.setter
    def slider_value(self, value: int):
        """Set a value (int) divided by self._divide_factor"""
        if value < self.minimum() or value > self.maximum():
            raise ValueError(
                f"The value for the slider ({value}) cannot be out of ({self.minimum()};{self.maximum()}) "
            )

        self.setValue(value)

        divided = value / self._divide_factor
        if self._divide_factor == 1.0:
            divided = int(divided)
        self._value_label.setText(str(divided))


class AnisotropyWidgets(QWidget):
    """Class that creates widgets for anisotropy handling. Includes :
    - A checkbox to hides or shows the controls
    - Three spinboxes to enter resolution for each dimension"""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        default_x: Optional[float] = 1.0,
        default_y: Optional[float] = 1.0,
        default_z: Optional[float] = 1.0,
        always_visible: Optional[bool] = False,
    ):
        """Creates an instance of AnisotropyWidgets
        Args:
            - parent: parent QWidget
            - default_x: default resolution to use for x axis in microns
            - default_y: default resolution to use for y axis in microns
            - default_z: default resolution to use for z axis in microns
        """
        super().__init__(parent)

        self._layout = QVBoxLayout()
        self._layout.setSpacing(0)
        self._layout.setContentsMargins(0, 0, 0, 0)

        self.container = ContainerWidget(t=7, parent=parent)
        self.checkbox = CheckBox(
            "Anisotropic data", self._toggle_display_aniso, parent
        )

        self.box_widgets = DoubleIncrementCounter.make_n(
            n=3, lower=1.0, upper=1000.0, default=1.0, step=0.5
        )
        self.box_widgets[0].setValue(default_x)
        self.box_widgets[1].setValue(default_y)
        self.box_widgets[2].setValue(default_z)

        for w in self.box_widgets:
            w.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.box_widgets_lbl = [
            make_label("Resolution in " + axis + " (microns) :", parent=parent)
            for axis in "xyz"
        ]

        ##################
        # tooltips
        self.checkbox.setToolTip(
            "If you have anisotropic data, you can scale data using your resolution in microns"
        )
        [
            w.setToolTip(f"Anisotropic resolution in microns for {dim} axis")
            for w, dim in zip(self.box_widgets, "xyz")
        ]
        ##################

        self.build()

        if always_visible:
            self._toggle_permanent_visibility()

    def _toggle_display_aniso(self):
        """Shows the choices for correcting anisotropy
        when viewing results depending on whether :py:attr:`self.checkbox` is checked"""
        toggle_visibility(self.checkbox, self.container)

    def build(self):
        """Builds the layout of the widget"""
        [
            self.container.layout.addWidget(widget, alignment=HCENTER_AL)
            for widgets in zip(self.box_widgets_lbl, self.box_widgets)
            for widget in widgets
        ]
        # anisotropy
        self.container.setLayout(self.container.layout)
        self.container.setVisible(False)

        add_widgets(self._layout, [self.checkbox, self.container])
        self.setLayout(self._layout)

    def resolution_xyz(self):
        """The resolution selected for each of the three dimensions. XYZ order"""
        return [w.value() for w in self.box_widgets]

    def scaling_xyz(self):
        """The scaling factors for each of the three dimensions. XYZ order"""
        return self.anisotropy_zoom_factor(self.resolution_xyz())

    def resolution_zyx(self):
        """The resolution selected for each of the three dimensions. ZYX order"""
        res = self.resolution_xyz()
        return [res[2], res[1], res[0]]

    def scaling_zyx(self):
        """The scaling factors for each of the three dimensions. XYZ order"""
        return self.anisotropy_zoom_factor(self.resolution_zyx())

    @staticmethod
    def anisotropy_zoom_factor(aniso_res):
        """Computes a zoom factor to correct anisotropy, based on anisotropy resolutions

            Args:
                aniso_res: array for anisotropic resolution (float) in microns for each axis

        Returns: an array with the corresponding zoom factors for each axis (all values divided by min)

        """

        base = min(aniso_res)
        zoom_factors = [base / res for res in aniso_res]
        return zoom_factors

    def enabled(self):
        """Returns : whether anisotropy correction has been enabled or not"""
        return self.checkbox.isChecked()

    def _toggle_permanent_visibility(self):
        """Hides the checkbox and always display resolution spinboxes"""
        self.checkbox.toggle()
        self.checkbox.setVisible(False)


class LayerSelecter(QWidget):
    def __init__(
        self, viewer, name="Layer", layer_type=napari.layers.Layer, parent=None
    ):

        super().__init__(parent)
        self._viewer = viewer

        self.image = None
        self.layer_type = layer_type

        self._container = ContainerWidget(fixed=False)

        self.layer_list = DropdownMenu(parent=self, label=name, fixed=False)
        # self.layer_list.setSizeAdjustPolicy(QComboBox.AdjustToContents) # use tooltip instead ?

        self._viewer.layers.events.inserted.connect(partial(self._add_layer))
        self._viewer.layers.events.removed.connect(partial(self._remove_layer))

        self.layer_list.currentIndexChanged.connect(self._update_tooltip)

        add_widgets(
            self._container.layout, [self.layer_list.label, self.layer_list]
        )
        self._check_for_layers()

    def _check_for_layers(self):

        for layer in self._viewer.layers:
            if isinstance(layer, self.layer_type):
                self.layer_list.addItem(layer.name)

    def _update_tooltip(self):

        self.layer_list.setToolTip(self.layer_list.currentText())

    def _add_layer(self, event):

        inserted_layer = event.value

        if isinstance(inserted_layer, self.layer_type):
            self.layer_list.addItem(inserted_layer.name)

    def _remove_layer(self, event):

        removed_layer = event.value

        if isinstance(
            removed_layer, self.layer_type
        ) and removed_layer.name in [
            self.layer_list.itemText(i) for i in range(self.layer_list.count())
        ]:

            index = self.layer_list.findText(removed_layer.name)
            self.layer_list.removeItem(index)

    def container(self):
        return self._container

    def layer_name(self):
        return self.layer_list.currentText()

    def layer_data(self):
        if self.layer_list.count() < 1:
            warnings.warn("Please select a valid layer !")
            return

        return self._viewer.layers[self.layer_name()].data


class FilePathWidget(QWidget):  # TODO include load as folder
    """Widget to handle the choice of file paths for data throughout the plugin. Provides the following elements :
    - An "Open" button to show a file dialog (defined externally)
    - A QLineEdit in read only to display the chosen path/file"""

    def __init__(
        self,
        description: str,
        file_function: callable,
        parent: Optional[QWidget] = None,
        required: Optional[bool] = True,
        default: Optional[str] = None,
    ):
        """Creates a FilePathWidget.
        Args:
            description (str): Initial text to add to the text box
            file_function (callable): Function to handle the file dialog
            parent (Optional[QWidget]): parent QWidget
            required (Optional[bool]): if True, field will be highlighted in red if empty. Defaults to False.
        """
        super().__init__(parent)
        self._layout = QHBoxLayout()
        self._layout.setSpacing(0)
        self._layout.setContentsMargins(0, 0, 0, 0)

        self._initial_desc = description
        self._text_field = QLineEdit(description, self)

        self._button = Button("Open", file_function, parent=self, fixed=True)

        self._text_field.setReadOnly(True)  # for user only
        if default is not None:
            self._text_field.setText(default)

        self._required = required

        self.build()
        self.check_ready()

    def build(self):
        """Builds the layout of the widget"""
        add_widgets(self._layout, [self.text_field, self.button])
        self.setLayout(self._layout)

    @property
    def text_field(self):
        """Get text field with file path"""
        return self._text_field

    @text_field.setter
    def text_field(self, text: str):
        """Sets the initial description in the text field, makes it the new default path"""
        self._initial_desc = text
        self._text_field.setText(text)

    @property
    def button(self):
        """Get "Open" button"""
        return self._button

    def check_ready(self):
        """Check if a path is correctly set"""
        if self.text_field.text() in ["", self._initial_desc]:
            self.update_field_color("indianred")
            self.text_field.setToolTip("Mandatory field !")
            return False
        else:
            self.update_field_color(f"{napari_param_darkgrey}")
            return True

    @property
    def required(self):
        return self._required

    @required.setter
    def required(self, is_required):
        """If set to True, will be colored red if incorrectly set"""
        if is_required:
            self.text_field.textChanged.connect(self.check_ready)
        else:
            try:
                self.text_field.textChanged.disconnect(self.check_ready)
            except TypeError:
                return
        self.check_ready()
        self._required = is_required

    def update_field_color(self, color: str):
        """Updates the background of the text field"""
        self.text_field.setStyleSheet(f"background-color : {color}")
        self.text_field.style().unpolish(self.text_field)
        self.text_field.style().polish(self.text_field)


class ScrollArea(QScrollArea):
    """Creates a QScrollArea and sets it up, then adds the contained_layout to it."""

    def __init__(
        self,
        contained_layout: QLayout,
        min_wh: Optional[List[int]] = None,
        max_wh: Optional[List[int]] = None,
        base_wh: Optional[List[int]] = None,
        parent: Optional[QWidget] = None,
    ):
        """
        Args:
              contained_layout (QLayout): the layout of widgets to be made scrollable
              min_wh (Optional[List[int]]): array of two ints for respectively the minimum width and minimum height of the scrollable area. Defaults to None, lets Qt decide if None
              max_wh (Optional[List[int]]): array of two ints for respectively the maximum width and maximum height of the scrollable area. Defaults to None, lets Qt decide if None
              base_wh (Optional[List[int]]): array of two ints for respectively the initial width and initial height of the scrollable area. Defaults to None, lets Qt decide if None
              parent (Optional[QWidget]): array of two ints for respectively the initial width and initial height of the scrollable area. Defaults to None, lets Qt decide if None
        """
        super().__init__(parent)

        self._container_widget = (
            QWidget()
        )  # required to use QScrollArea.setWidget()
        self._container_widget.setSizePolicy(
            QSizePolicy.Fixed, QSizePolicy.Maximum
        )
        self._container_widget.setLayout(contained_layout)
        self._container_widget.adjustSize()

        self.setWidget(self._container_widget)
        self.setWidgetResizable(True)
        self.setSizePolicy(
            QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding
        )

        if base_wh is not None:
            self.setBaseSize(base_wh[0], base_wh[1])
        if max_wh is not None:
            self.setMaximumSize(max_wh[0], max_wh[1])
        if min_wh is not None:
            self.setMinimumSize(min_wh[0], min_wh[1])

        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

    @classmethod
    def make_scrollable(
        cls,
        contained_layout: QLayout,
        parent: QWidget,
        min_wh: Optional[List[int]] = None,
        max_wh: Optional[List[int]] = None,
        base_wh: Optional[List[int]] = None,
    ):
        """Factory method to create a scroll area in a widget
        Args:
                contained_layout (QLayout): the widget to be made scrollable
                parent (QWidget): the parent widget to add the resulting scroll area in
                min_wh (Optional[List[int]]): array of two ints for respectively the minimum width and minimum height of the scrollable area. Defaults to None, lets Qt decide if None
                max_wh (Optional[List[int]]): array of two ints for respectively the maximum width and maximum height of the scrollable area. Defaults to None, lets Qt decide if None
                base_wh (Optional[List[int]]): array of two ints for respectively the initial width and initial height of the scrollable area. Defaults to None, lets Qt decide if None
        """

        scroll = cls(contained_layout, min_wh, max_wh, base_wh)
        layout = QVBoxLayout(parent)
        # layout.setContentsMargins(0,0,1,1)
        layout.setSizeConstraint(QLayout.SetMinAndMaxSize)
        layout.addWidget(scroll)
        parent.setLayout(layout)


def set_spinbox(
    box,
    min=0,
    max=10,
    default=0,
    step=1,
    fixed: Optional[bool] = True,
):
    """Args:
    box : QSpinBox or QDoubleSpinBox
    min : minimum value, defaults to 0
    max : maximum value, defaults to 10
    default :  default value, defaults to 0
    step : step value, defaults to 1
    fixed (bool): if True, sets the QSizePolicy of the spinbox to Fixed"""

    box.setMinimum(min)
    box.setMaximum(max)
    box.setSingleStep(step)
    box.setValue(default)

    if fixed:
        box.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)


def make_n_spinboxes(
    class_,
    n: int = 2,
    min=0,
    max=10,
    default=0,
    step=1,
    parent: Optional[QWidget] = None,
    fixed: Optional[bool] = True,
):
    """Creates n increment counters with the specified parameters :

    Args:
        class_ : QSpinBox or QDoubleSpinbox
        n (int): number of increment counters to create
        min (Optional[int]): minimum value, defaults to 0
        max (Optional[int]): maximum value, defaults to 10
        default (Optional[int]): default value, defaults to 0
        step (Optional[int]): step value, defaults to 1
        parent: parent widget, defaults to None
        fixed (bool): if True, sets the QSizePolicy of the spinbox to Fixed
    """
    if n <= 1:
        raise ValueError("Cannot make less than 2 spin boxes")

    boxes = []
    for i in range(n):
        box = class_(min, max, default, step, parent, fixed)
        boxes.append(box)
    return boxes


class DoubleIncrementCounter(QDoubleSpinBox):
    """Class implementing a number counter with increments (spin box) for floats."""

    def __init__(
        self,
        lower: Optional[float] = 0.0,
        upper: Optional[float] = 10.0,
        default: Optional[float] = 0.0,
        step: Optional[float] = 1.0,
        parent: Optional[QWidget] = None,
        fixed: Optional[bool] = True,
        label: Optional[str] = None,
    ):
        """Args:
        lower (Optional[float]): minimum value, defaults to 0
        upper (Optional[float]): maximum value, defaults to 10
        default (Optional[float]): default value, defaults to 0
        step (Optional[float]): step value, defaults to 1
        parent: parent widget, defaults to None
        fixed (bool): if True, sets the QSizePolicy of the spinbox to Fixed
        label (Optional[str]): if provided, creates a label with the chosen title to use with the counter"""

        super().__init__(parent)
        set_spinbox(self, lower, upper, default, step, fixed)

        self.layout = None

        if label is not None:
            self.label = make_label(name=label)

    @property
    def tooltips(self):
        return self.toolTip()

    @tooltips.setter
    def tooltips(self, tooltip: str):
        """Sets the tooltip of both the DoubleIncrementCounter and its label"""
        self.setToolTip(tooltip)
        if self.label is not None:
            self.label.setToolTip(tooltip)

    @property
    def precision(self):
        return self.decimals()

    @precision.setter
    def precision(self, decimals: int):
        """Sets the precision of the box to the specified number of decimals"""
        self.setDecimals(decimals)

    @classmethod
    def make_n(
        cls,
        n: int = 2,
        lower: float = 0,
        upper: float = 10,
        default: float = 0,
        step: float = 1,
        parent: Optional[QWidget] = None,
        fixed: Optional[bool] = True,
    ):
        return make_n_spinboxes(
            cls, n, lower, upper, default, step, parent, fixed
        )


class IntIncrementCounter(QSpinBox):
    """Class implementing a number counter with increments (spin box) for int."""

    def __init__(
        self,
        lower=0,
        upper=10,
        default=0,
        step=1,
        parent: Optional[QWidget] = None,
        fixed: Optional[bool] = True,
        label: Optional[str] = None,
    ):
        """Args:
        lower (Optional[int]): minimum value, defaults to 0
        upper (Optional[int]): maximum value, defaults to 10
        default (Optional[int]): default value, defaults to 0
        step (Optional[int]): step value, defaults to 1
        parent: parent widget, defaults to None
        fixed (bool): if True, sets the QSizePolicy of the spinbox to Fixed"""

        super().__init__(parent)
        set_spinbox(self, lower, upper, default, step, fixed)

        self.label = None
        self.container = None

        if label is not None:
            self.label = make_label(name=label)

    @property
    def tooltips(self):
        return self.toolTip()

    @tooltips.setter
    def tooltips(self, tooltip):
        self.setToolTip(tooltip)
        self.label.setToolTip(tooltip)

    @classmethod
    def make_n(
        cls,
        n: int = 2,
        lower: int = 0,
        upper: int = 10,
        default: int = 0,
        step: int = 1,
        parent: Optional[QWidget] = None,
        fixed: Optional[bool] = True,
    ):
        return make_n_spinboxes(
            cls, n, lower, upper, default, step, parent, fixed
        )


def add_blank(widget, layout=None):
    """
    Adds a space between consecutive buttons/labels in a layout when building a widget

    Args:
        widget (QWidget): widget to add blank in
        layout (QLayout): layout to add blank in

    Returns:
        QLabel : blank label
    """
    blank = QLabel("", widget)
    if layout is not None:
        layout.addWidget(blank, alignment=ABS_AL)
    return blank


def open_file_dialog(
    widget,
    possible_paths: list = [],
    filetype: str = "Image file (*.tif *.tiff)",
):
    """Opens a window to choose a file directory using QFileDialog.

    Args:
        widget (QWidget): Widget to display file dialog in
        possible_paths (str): Paths that may have been chosen before, can be a string
        or an array of strings containing the paths
        load_as_folder (bool): Whether to open a folder or a single file. If True, will allow opening folder as a single file (2D stack interpreted as 3D)
        filetype (str): The description and file extension to load (format : ``"Description (*.example1 *.example2)"``). Default ``"Image file (*.tif *.tiff)"``

    """

    default_path = utils.parse_default_path(possible_paths)

    f_name = QFileDialog.getOpenFileName(
        widget, "Choose file", default_path, filetype
    )
    return f_name


def open_folder_dialog(
    widget,
    possible_paths: list = [],
):
    default_path = utils.parse_default_path(possible_paths)

    print(f"Default : {default_path}")
    filenames = QFileDialog.getExistingDirectory(
        widget, "Open directory", default_path
    )
    return filenames


def make_label(name, parent=None):  # TODO update to child class
    """Creates a QLabel

    Args:
        name: string with name
        parent: parent widget

    Returns: created label

    """
    if parent is not None:
        return QLabel(name, parent)
    else:
        return QLabel(name)


def make_group(title, l=7, t=20, r=7, b=11, parent=None):
    """Creates a group widget and layout, with a header (`title`) and content margins for top/left/right/bottom `L, T, R, B` (in pixels)
    Group widget and layout returned will have a Fixed size policy.

    Args:
        title (str): Title of the group
        l (int): left margin
        t (int): top margin
        r (int): right margin
        b (int): bottom margin
        parent (QWidget) : parent widget. If None, no parent is set
    """
    group = GroupedWidget(title, l, t, r, b, parent=parent)
    layout = group.layout

    return group, layout


class GroupedWidget(QGroupBox):
    """Subclass of QGroupBox designed to easily group widgets belonging to a same category"""

    def __init__(self, title, l=7, t=20, r=7, b=11, parent=None):
        super().__init__(title, parent)

        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(l, t, r, b)
        self.layout.setSizeConstraint(QLayout.SetFixedSize)

    @classmethod
    def create_single_widget_group(
        cls, title, widget, layout, l=7, t=20, r=7, b=11
    ):
        group = cls(title, l, t, r, b)
        group.layout.addWidget(widget)
        group.setLayout(group.layout)
        layout.addWidget(group)



class GroupedWidget(QGroupBox):
    """Subclass of QGroupBox designed to easily group widgets belonging to a same category"""

    def __init__(self, title, l=7, t=20, r=7, b=11, parent=None):
        super().__init__(title, parent)

        self.setSizePolicy(QSizePolicy.Fixed)

        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(l, t, r, b)
        self.layout.setSizeConstraint(QLayout.SetFixedSize)

    @property
    def layout(self) -> "QLayout":
        return self.layout

    @classmethod
    def create_single_widget_group(
        cls, title, widget, layout, l=7, t=20, r=7, b=11
    ):
        group, group_layout = cls(title, l, t, r, b)
        group_layout.addWidget(widget)
        group.setLayout(group_layout)
        layout.addWidget(group)


class ContainerWidget(QWidget):  # TODO convert calls
    def __init__(self, l=0, t=0, r=1, b=11, vertical=True, parent=None):

        super().__init__(parent)
        self.layout = None

        if vertical:
            self.layout = QVBoxLayout(self)
        else:
            self.layout = QHBoxLayout(self)

        self.layout.setContentsMargins(l, t, r, b)
        if fixed:
            self.layout.setSizeConstraint(QLayout.SetFixedSize)


def add_widgets(layout, widgets, alignment=LEFT_AL):
    """Adds all widgets in the list to layout, with the specified alignment.
    If alignment is None, no alignment is set.
    Args:
        layout: layout to add widgets in
        widgets: list of QWidgets to add to layout
        alignment: any valid Qt.AlignmentFlag, see aliases at beginning of interface.py. If None, uses default of addWidget
    """
    if alignment is None:
        for w in widgets:
            layout.addWidget(w)
    else:
        for w in widgets:
            layout.addWidget(w, alignment=alignment)


def combine_blocks(  # TODO FIXME PLEASE
    right_or_below,
    left_or_above,
    min_spacing=0,
    horizontal=True,
    l=11,
    t=3,
    r=11,
    b=11,
):
    """Combines two QWidget objects and puts them side by side (first on the left/top and second on the right/bottom depending on "horizontal")
       Weird argument names due the initial implementation of it.  # TODO maybe fix arg names

    Args:
        left_or_above (QWidget): First widget, to be added on the left/above of "second"
        right_or_below (QWidget): Second widget, to be displayed right/below of "first"
        min_spacing (int): Minimum spacing between the two widgets (from the start of label to the start of button)
                horizontal (bool): whether to stack widgets vertically (False) or horizontally (True)
        l (int): left spacing in pixels
        t (int): top spacing in pixels
        r (int): right spacing in pixels
        b (int): bottom spacing in pixels

    Returns:
        QWidget: new QWidget containing the merged widget and label
    """
    temp_widget = QWidget()
    temp_widget.setSizePolicy(
        QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding
    )

    temp_layout = QGridLayout()
    if horizontal:
        temp_widget.setSizePolicy(
            QSizePolicy.MinimumExpanding, QSizePolicy.Maximum
        )
        temp_layout.setColumnMinimumWidth(0, min_spacing)
        c1, c2, r1, r2 = 0, 1, 0, 0
        temp_layout.setContentsMargins(
            l, t, r, b
        )  # determines spacing between widgets
    else:
        temp_widget.setSizePolicy(
            QSizePolicy.Maximum, QSizePolicy.MinimumExpanding
        )
        temp_layout.setRowMinimumHeight(0, min_spacing)
        c1, c2, r1, r2 = 0, 0, 0, 1
        temp_layout.setContentsMargins(
            l, t, r, b
        )  # determines spacing between widgets
    # temp_layout.setColumnMinimumWidth(1,100)
    # temp_layout.setSizeConstraint(QLayout.SetMinAndMaxSize)

    temp_layout.addWidget(left_or_above, r1, c1)  # , alignment=LEFT_AL)
    # temp_layout.addStretch(100)
    temp_layout.addWidget(right_or_below, r2, c2)  # , alignment=LEFT_AL)
    temp_widget.setLayout(temp_layout)
    return temp_widget


def open_url(url):
    """Opens the url given as a string in OS default browser using :py:func:`QDesktopServices.openUrl`.

    Args:
        url (str): Url to be opened
    """
    QDesktopServices.openUrl(QUrl(url, QUrl.TolerantMode))
