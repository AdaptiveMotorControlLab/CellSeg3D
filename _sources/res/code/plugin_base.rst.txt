plugin_base.py
====================================================


Class : BasePluginSingleImage
----------------------------------------------------


Methods
**********************
.. autoclass:: napari_cellseg3d.code_plugins.plugin_base::BasePluginSingleImage
   :members:  __init__, enable_utils_menu, remove_from_viewer, remove_docked_widgets
   :noindex:

Attributes
*********************

.. autoclass:: napari_cellseg3d.code_plugins.plugin_base::BasePluginSingleImage
   :members:  _viewer, image_path, label_path, image_layer_loader, label_layer_loader




Class : BasePluginFolder
-------------------------------------------------------


Methods
***********************
.. autoclass:: napari_cellseg3d.code_plugins.plugin_base::BasePluginFolder
   :members:  __init__, load_dataset_paths,load_image_dataset,load_label_dataset
   :noindex:

Attributes
*********************

.. autoclass:: napari_cellseg3d.code_plugins.plugin_base::BasePluginFolder
   :members:  _viewer, images_filepaths, labels_filepaths, results_path
