plugin_model_training.py
====================================================


Class : Trainer
----------------------------------------------------

.. important::
    Inherits from : :doc:`model_framework`

Methods
**********************
.. autoclass:: napari_cellseg_annotator.plugin_model_training::Trainer
   :members:  __init__, build, show_dialog_lab, show_dialog_dat, check_ready, start, on_start, on_finish, on_error, on_yield, plot_loss, update_loss_plot, close
   :noindex:



Attributes
*********************

.. autoclass:: napari_cellseg_annotator.plugin_model_training::Trainer
   :members:  _viewer, worker, models_dict, loss_dict, canvas, train_loss_plot, dice_metric_plot