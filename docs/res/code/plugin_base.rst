plugin_base.py
====================================================


Class : BasePluginSingleImage
----------------------------------------------------


Methods
**********************
.. autoclass:: napari_cellseg3d.plugin_base::BasePluginSingleImage
   :members:  __init__, remove_from_viewer, show_dialog_images, show_dialog_labels, update_default
   :noindex:



Attributes
*********************

.. autoclass:: napari_cellseg3d.plugin_base::BasePluginSingleImage
   :members:  _viewer, image_path, label_path, filetype, file_handling_box




Class : BasePluginFolder
-------------------------------------------------------


Methods
***********************
.. autoclass:: napari_cellseg3d.plugin_base::BasePluginFolder
   :members:  __init__, remove_from_viewer,make_close_button,make_prev_button,make_next_button, load_dataset_paths,load_image_dataset,load_label_dataset,load_results_path, update_default,remove_docked_widgets
   :noindex:


Attributes
*********************

.. autoclass:: napari_cellseg3d.plugin_base::BasePluginFolder
   :members:  _viewer, images_filepaths, labels_filepaths, results_path