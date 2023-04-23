model_workers.py
===========================================


Class : LogSignal
-------------------------------------------

.. important::
    Inherits from :py:class:`napari.qt.threading.WorkerBaseSignals`

Attributes
************************
.. autoclass:: napari_cellseg3d.code_models.model_workers::LogSignal
   :members: log_signal
   :noindex:



Class : InferenceWorker
-------------------------------------------

.. important::
    Inherits from :py:class:`napari.qt.threading.GeneratorWorker`

Methods
************************
.. autoclass:: napari_cellseg3d.code_models.model_workers::InferenceWorker
   :members: __init__, log, create_inference_dict, inference
   :noindex:

.. _here: https://napari-staging-site.github.io/guides/stable/threading.html


Class : TrainingWorker
-------------------------------------------

.. important::
    Inherits from :py:class:`napari.qt.threading.GeneratorWorker`

Methods
************************
.. autoclass:: napari_cellseg3d.code_models.model_workers::TrainingWorker
   :members: __init__, log, train
   :noindex:
