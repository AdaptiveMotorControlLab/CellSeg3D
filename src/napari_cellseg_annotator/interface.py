from qtpy.QtCore import Qt
from qtpy.QtCore import QUrl
from qtpy.QtGui import QDesktopServices
from qtpy.QtWidgets import QFileDialog
from qtpy.QtWidgets import QGridLayout
from qtpy.QtWidgets import QGroupBox
from qtpy.QtWidgets import QLabel
from qtpy.QtWidgets import QLayout
from qtpy.QtWidgets import QPushButton
from qtpy.QtWidgets import QScrollArea
from qtpy.QtWidgets import QSizePolicy
from qtpy.QtWidgets import QSpinBox
from qtpy.QtWidgets import QVBoxLayout
from qtpy.QtWidgets import QWidget

from napari_cellseg_annotator import utils

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
    widget, possible_paths=[""], load_as_folder: bool = False
):
    """Opens a window to choose a file directory using QFileDialog.

    Args:
        widget (QWidget): Widget to display file dialog in
        possible_paths (str): Paths that may have been chosen before, can be a string
        or an array of strings containing the paths
        load_as_folder (bool): Whether to open a folder or a single file. If True, will allow to open folder as a single file (2D stack interpreted as 3D)
    """

    default_path = utils.parse_default_path(possible_paths)
    if not load_as_folder:
        f_name = QFileDialog.getOpenFileName(
            widget, "Choose file", default_path, "Image file (*.tif *.tiff)"
        )
        return f_name
    else:
        print(default_path)
        filenames = QFileDialog.getExistingDirectory(
            widget, "Open directory", default_path
        )
        return filenames


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


def make_n_spinboxes(n=1, min=0, max=10, default=0, step=1, parent=None):
    """

    Args:
        n: number of spinboxes, defaults to 1
        min: min value, defaults to 0
        max: max value, defaults to 10
        default: default value, defaults to 0
        step : step value, defaults to 1
        parent: parent widget, defaults to None

    Returns:
            list: A list of n QSpinBoxes with specified parameters. If only one box is made, returns the box itself instead
    """
    boxes = []
    for i in range(n):
        if parent is not None:
            widget = QSpinBox(parent)
        else:
            widget = QSpinBox()
        widget.setMinimum(min)
        widget.setMaximum(max)
        widget.setSingleStep(step)
        widget.setValue(default)
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


def make_container_widget(L=0, T=0, R=1, B=11):
    """Creates a QWidget and a layout for the purpose of containing other modules, with a Fixed layout.

    Args:
        L: left margin of layout
        T: top margin of layout
        R: right margin of layout
        B: bottom margin of layout

    Returns:
        QWidget : widget that contains the other widgets. Fixed size.
        QVBoxLayout :  layout to add contained widgets in. Fixed size.
    """
    container_widget = QWidget()
    container_layout = QVBoxLayout()
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
