workers.py
===========================================


Class : LogSignal
-------------------------------------------

.. important::
    Inherits from :py:class:`napari.qt.threading.WorkerBaseSignals`

Attributes
************************
.. autoclass:: napari_cellseg3d.code_models.workers_utils::LogSignal
   :members: log_signal
   :noindex:



Class : InferenceWorker
-------------------------------------------

.. important::
    Inherits from :py:class:`napari.qt.threading.GeneratorWorker`

Methods
************************
.. autoclass:: napari_cellseg3d.code_models.worker_inference::InferenceWorker
   :members: __init__, log, create_inference_dict, inference
   :noindex:

.. _here: https://napari-staging-site.github.io/guides/stable/threading.html


Class : TrainingWorkerBase
-------------------------------------------

.. important::
    Inherits from :py:class:`napari.qt.threading.GeneratorWorker`

Methods
************************
.. autoclass:: napari_cellseg3d.code_models.worker_training::TrainingWorkerBase
   :members: __init__, log, train
   :noindex:


Class : WNetTrainingWorker
-------------------------------------------

.. important::
    Inherits from :py:class:`TrainingWorkerBase`

Methods
************************
.. autoclass:: napari_cellseg3d.code_models.worker_training::WNetTrainingWorker
   :members: __init__, train, eval, get_patch_dataset, get_dataset_eval, get_dataset
   :noindex:


Class : SupervisedTrainingWorker
-------------------------------------------

.. important::
    Inherits from :py:class:`TrainingWorkerBase`

Methods
************************
.. autoclass:: napari_cellseg3d.code_models.worker_training::SupervisedTrainingWorker
   :members: __init__, train
   :noindex:
