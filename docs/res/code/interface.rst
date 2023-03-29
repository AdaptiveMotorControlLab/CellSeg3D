interface.py
=============

Classes
-------------

QWidgetSingleton
**************************************
.. autoclass:: napari_cellseg3d.interface::QWidgetSingleton
    :members: __call__

UtilsDropdown
**************************************
.. autoclass:: napari_cellseg3d.interface::UtilsDropdown
    :members: __init__, dropdown_menu_call, show_utils_menu

Log
**************************************
.. autoclass:: napari_cellseg3d.interface::Log
    :members: __init__, write, replace_last_line, print_and_log, warn


ContainerWidget
**************************************
.. autoclass:: napari_cellseg3d.interface::ContainerWidget
    :members: __init__

Button
**************************************
.. autoclass:: napari_cellseg3d.interface::Button
   :members: __init__, visibility_condition

DropdownMenu
**************************************
.. autoclass:: napari_cellseg3d.interface::DropdownMenu
   :members: __init__

CheckBox
**************************************
.. autoclass:: napari_cellseg3d.interface::CheckBox
   :members: __init__

AnisotropyWidgets
**************************************
.. autoclass:: napari_cellseg3d.interface::AnisotropyWidgets
   :members: __init__, build, scaling_zyx, resolution_zyx, scaling_xyz, resolution_xyz,enabled


FilePathWidget
**************************************
.. autoclass:: napari_cellseg3d.interface::FilePathWidget
   :members: __init__, build, text_field, button, check_ready, required, update_field_color, tooltips

ScrollArea
**************************************
.. autoclass:: napari_cellseg3d.interface::ScrollArea
   :members: __init__, make_scrollable

DoubleIncrementCounter
**************************************
.. autoclass:: napari_cellseg3d.interface::DoubleIncrementCounter
   :members: __init__, precision, make_n

IntIncrementCounter
**************************************
.. autoclass:: napari_cellseg3d.interface::IntIncrementCounter
   :members: __init__, make_n


Functions
-------------

handle_adjust_errors
**************************************
.. autofunction:: napari_cellseg3d.interface::handle_adjust_errors

handle_adjust_errors_wrapper
**************************************
.. autofunction:: napari_cellseg3d.interface::handle_adjust_errors_wrapper

open_url
**************************************
.. autofunction:: napari_cellseg3d.interface::open_url

make_group
**************************************
.. autofunction:: napari_cellseg3d.interface::make_group

combine_blocks
**************************************
.. autofunction:: napari_cellseg3d.interface::combine_blocks

add_blank
**************************************
.. autofunction:: napari_cellseg3d.interface::add_blank

add_label
**************************************
.. autofunction:: napari_cellseg3d.interface::add_label

toggle_visibility
**************************************
.. autofunction:: napari_cellseg3d.interface::toggle_visibility

open_file_dialog
**************************************
.. autofunction:: napari_cellseg3d.interface::open_file_dialog



















