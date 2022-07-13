interface.py
=============

Classes
-------------

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
   :members: __init__, build, get_anisotropy_resolution_xyz, get_anisotropy_resolution_zyx, anisotropy_zoom_factor,is_enabled,toggle_permanent_visibility


FilePathWidget
**************************************
.. autoclass:: napari_cellseg3d.interface::FilePathWidget
   :members: __init__, build, get_text_field, get_button, check_ready, set_required, update_field_color, set_description

ScrollArea
**************************************
.. autoclass:: napari_cellseg3d.interface::ScrollArea
   :members: __init__, make_scrollable

DoubleIncrementCounter
**************************************
.. autoclass:: napari_cellseg3d.interface::DoubleIncrementCounter
   :members: __init__, set_precision, make_n

IntIncrementCounter
**************************************
.. autoclass:: napari_cellseg3d.interface::IntIncrementCounter
   :members: __init__, make_n


Functions
-------------

open_url
**************************************
.. autofunction:: napari_cellseg3d.interface::open_url


make_group
**************************************
.. autofunction:: napari_cellseg3d.interface::make_group

add_to_group
**************************************
.. autofunction:: napari_cellseg3d.interface::add_to_group

make_container
**************************************
.. autofunction:: napari_cellseg3d.interface::make_container

combine_blocks
**************************************
.. autofunction:: napari_cellseg3d.interface::combine_blocks

add_blank
**************************************
.. autofunction:: napari_cellseg3d.interface::add_blank

toggle_visibility
**************************************
.. autofunction:: napari_cellseg3d.interface::toggle_visibility

open_file_dialog
**************************************
.. autofunction:: napari_cellseg3d.interface::open_file_dialog



















