plugin_model_training.py
====================================================


Class : Trainer
----------------------------------------------------

.. important::
    Inherits from : :doc:`model_framework`

Methods
**********************
.. autoclass:: napari_cellseg3d.code_plugins.plugin_model_training::Trainer
   :members:  __init__, get_loss, check_ready, send_log, start, on_start, on_finish, on_error, on_yield, plot_loss, update_loss_plot
   :noindex:



Attributes
*********************

.. autoclass:: napari_cellseg3d.code_plugins.plugin_model_training::Trainer
   :members:  _viewer, worker, loss_dict, canvas, train_loss_plot, dice_metric_plot