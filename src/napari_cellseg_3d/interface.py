from qtpy.QtCore import QUrl
from qtpy.QtCore import Qt
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
from qtpy.QtWidgets import QPushButton
from qtpy.QtWidgets import QScrollArea
from qtpy.QtWidgets import QSizePolicy
from qtpy.QtWidgets import QSpinBox
from qtpy.QtWidgets import QVBoxLayout
from qtpy.QtWidgets import QWidget

from napari_cellseg_3d import utils

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

dark_red = "#72071d"  # crimson red
default_cyan = "#8dd3c7"  # turquoise cyan (default matplotlib line color under dark background context)
napari_grey = "#262930"  # napari background color (grey)


def add_blank(widget, layout):
    """
    Adds a space between consecutive buttons/labels in a layout when building a widget

    Args:
        widget (QWidget): widget to add blank in
        layout (QLayout): layout to add blank in

    Returns:
        QLabel : blank label
    """
    blank = QLabel("", widget)
    layout.addWidget(blank, alignment=ABS_AL)
    return blank


def open_file_dialog(
    widget,
    possible_paths=[""],
    load_as_folder: bool = False,
    filetype: str = "Image file (*.tif *.tiff)",
):
    """Opens a window to choose a file directory using QFileDialog.

    Args:
        widget (QWidget): Widget to display file dialog in
        possible_paths (str): Paths that may have been chosen before, can be a string
        or an array of strings containing the paths
        load_as_folder (bool): Whether to open a folder or a single file. If True, will allow to open folder as a single file (2D stack interpreted as 3D)
        filetype (str): The description and file extension to load (format : "Description (*.example1 *.example2)"). Default "Image file (*.tif *.tiff)"

    """

    default_path = utils.parse_default_path(possible_paths)
    if not load_as_folder:
        f_name = QFileDialog.getOpenFileName(
            widget, "Choose file", default_path, filetype
        )
        return f_name
    else:
        print(default_path)
        filenames = QFileDialog.getExistingDirectory(
            widget, "Open directory", default_path
        )
        return filenames


def make_label(name, parent):
    """Creates a QLabel

    Args:
        name: string with name
        parent: parent widget

    Returns: created label

    """
    return QLabel(name, parent)


def make_scrollable(
    contained_layout, containing_widget, min_wh=None, max_wh=None, base_wh=None
):
    """Creates a QScrollArea and sets it up, then adds the contained_widget to it,
    and finally adds the scroll area in a layout and sets it to the contaning_widget


    Args:
        contained_layout (QLayout): the widget to be made scrollable
        containing_widget (QWidget): the widget to add the resulting scroll area in
        min_wh (array(int)): array of two ints for respectively the minimum width and minimum height of the scrollable area. Defaults to None, lets Qt decide if None
        max_wh (array(int)): array of two ints for respectively the maximum width and maximum height of the scrollable area. Defaults to None, lets Qt decide if None
        base_wh (array(int)): array of two ints for respectively the initial width and initial height of the scrollable area. Defaults to None, lets Qt decide if None
    """
    container_widget = QWidget()  # required to use QScrollArea.setWidget()
    container_widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Maximum)
    container_widget.setLayout(contained_layout)
    container_widget.adjustSize()
    # TODO : could we optimize the number of created objects ?
    scroll = QScrollArea()
    scroll.setWidget(container_widget)
    scroll.setWidgetResizable(True)
    scroll.setSizePolicy(
        QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding
    )
    if base_wh is not None:
        scroll.setBaseSize(base_wh[0], base_wh[1])
    if max_wh is not None:
        scroll.setMaximumSize(max_wh[0], max_wh[1])
    if min_wh is not None:
        scroll.setMinimumSize(min_wh[0], min_wh[1])

    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    # scroll.adjustSize()

    layout = QVBoxLayout(containing_widget)
    # layout.setContentsMargins(0,0,1,1)
    layout.setSizeConstraint(QLayout.SetMinAndMaxSize)
    layout.addWidget(scroll)
    containing_widget.setLayout(layout)


def make_n_spinboxes(
    n=1,
    min=0,
    max=10,
    default=0,
    step=1,
    parent=None,
    double=False,
    fixed=True,
):
    """

    Args:
        n: number of spinboxes, defaults to 1
        min: min value, defaults to 0
        max: max value, defaults to 10
        default: default value, defaults to 0
        step : step value, defaults to 1
        parent: parent widget, defaults to None
        double (bool): if True, creates a QDoubleSpinBox rather than a QSpinBox
        fixed (bool): if True, sets the QSizePolicy of the spinbox to Fixed

    Returns:
            list: A list of n Q(Double)SpinBoxes with specified parameters. If only one box is made, returns the box itself instead
    """
    if double:
        box_type = QDoubleSpinBox
    else:
        box_type = QSpinBox
    boxes = []
    for i in range(n):
        if parent is not None:
            widget = box_type(parent)
        else:
            widget = box_type()
        widget.setMinimum(min)
        widget.setMaximum(max)
        widget.setSingleStep(step)
        widget.setValue(default)

        if fixed:
            widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        boxes.append(widget)
    if len(boxes) == 1:
        return boxes[0]
    return boxes


def make_group(title, L=7, T=20, R=7, B=11, solo_dict=None):
    """Creates a group with a header (`title`) and content margins for top/left/right/bottom `L, T, R, B` (in pixels)
    Group widget and layout returned will have a Fixed size policy.
    If solo_dict is not None, adds specified widget to specified layout and returns None.

    Args:
        title (str): Title of the group
        L (int): left margin
        T (int): top margin
        R (int): right margin
        B (int): bottom margin
        solo_dict (dict): shortcut if only one widget is to be added to the group. Should contain "widget" (QWidget) and "layout" (Qlayout), widget will be added to layout. Defaults to None
    """
    group = QGroupBox(title)
    group.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
    layout = QVBoxLayout()
    layout.setContentsMargins(L, T, R, B)
    layout.setSizeConstraint(QLayout.SetFixedSize)

    if (
        solo_dict is not None
    ):  # use the dict to directly add a widget if it is alone in the group
        external_lay = solo_dict["layout"]
        external_wid = solo_dict["widget"]
        layout.addWidget(external_wid)  # , alignment=LEFT_AL)
        group.setLayout(layout)
        external_lay.addWidget(group)
        return

    return group, layout


def make_container_widget(L=0, T=0, R=1, B=11, vertical=True):
    """Creates a QWidget and a layout for the purpose of containing other modules, with a Fixed layout.

    Args:
        L (int): left margin of layout
        T (int): top margin of layout
        R (int): right margin of layout
        B (int): bottom margin of layout
        vertical (bool): if False, uses QHBoxLayout instead of QVboxLayout. Default: True

    Returns:
        QWidget : widget that contains the other widgets. Fixed size.
        QBoxLayout :  H/V Box layout to add contained widgets in. Fixed size.
    """
    container_widget = QWidget()

    if vertical:
        container_layout = QVBoxLayout()
    else:
        container_layout = QHBoxLayout()
    container_layout.setContentsMargins(L, T, R, B)
    container_layout.setSizeConstraint(QLayout.SetFixedSize)

    return container_widget, container_layout


def make_button(
    title: str = None,
    func: callable = None,
    parent: QWidget = None,
    fixed: bool = True,
):
    """Creates a button with a title and connects it to a function when clicked

    Args:
        title (str-like): title of the button. Defaults to None, if None no title is set
        func (callable): function to execute when button is clicked. Defaults to None, no binding is made if None
        parent (QWidget): parent QWidget to add button to. Defaults to None, no parent is set if None
        fixed (bool): if True, will set the size policy of the button to Fixed in h and w. Defaults to True.

    Returns:
        QPushButton : created button
    """
    if parent is not None:
        if title is not None:
            btn = QPushButton(title, parent)
        else:
            btn = QPushButton(parent)
    else:
        if title is not None:
            btn = QPushButton(title, parent)
        else:
            btn = QPushButton(parent)

    if fixed:
        btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    if func is not None:
        btn.clicked.connect(func)

    return btn


def make_combobox(
    entries=None,
    parent: QWidget = None,
    label: str = None,
    fixed: bool = True,
):
    """Creates a dropdown menu with a title and adds specified entries to it

    Args:
        entries array(str): Entries to add to the dropdown menu. Defaults to None, no entries if None
        parent (QWidget): parent QWidget to add dropdown menu to. Defaults to None, no parent is set if None
        label (str) : if not None, creates a Qlabel with the contents of 'label', and returns the label as well
        fixed (bool): if True, will set the size policy of the dropdown menu to Fixed in h and w. Defaults to True.

    Returns:
        QComboBox : created dropdown menu
    """
    if parent is None:
        menu = QComboBox()
    else:
        menu = QComboBox(parent)

    if entries is not None:
        menu.addItems(entries)

    if fixed:
        menu.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    if label is not None:
        label = QLabel(label)
        return menu, label

    return menu


def make_checkbox(
    title: str = None,
    func: callable = None,
    parent: QWidget = None,
    fixed: bool = True,
):
    """Creates a checkbox with a title and connects it to a function when clicked

    Args:
        title (str-like): title of the checkbox. Defaults to None, if None no title is set
        func (callable): function to execute when checkbox is toggled. Defaults to None, no binding is made if None
        parent (QWidget): parent QWidget to add checkbox to. Defaults to None, no parent is set if None
        fixed (bool): if True, will set the size policy of the checkbox to Fixed in h and w. Defaults to True.

    Returns:
        QCheckBox : created button
    """
    if parent is not None:
        if title is not None:
            box = QCheckBox(title, parent)
        else:
            box = QCheckBox(parent)
    else:
        if title is not None:
            box = QCheckBox(title, parent)
        else:
            box = QCheckBox(parent)

    if fixed:
        box.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    if func is not None:
        box.toggled.connect(func)

    return box


def combine_blocks(
    second, first, min_spacing=0, horizontal=True, l=11, t=3, r=11, b=11
):
    """Combines two QWidget objects and puts them side by side (label on the left and button on the right)

    Args:
        horizontal (bool): whether to stack widgets laterally or horizontally
        second (QWidget): Second widget, to be displayed right/below of the label
        first (QWidget): First widget, to be added on the left/above of button
        min_spacing (int): Minimum spacing between the two widgets (from the start of label to the start of button)

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

    temp_layout.addWidget(first, r1, c1)  # , alignment=LEFT_AL)
    # temp_layout.addStretch(100)
    temp_layout.addWidget(second, r2, c2)  # , alignment=LEFT_AL)
    temp_widget.setLayout(temp_layout)
    return temp_widget


def open_url(url):
    """Opens the url given as a string in OS default browser using :py:func:`QDesktopServices.openUrl`.

    Args:
        url (str): Url to be opened
    """
    QDesktopServices.openUrl(QUrl(url, QUrl.TolerantMode))
