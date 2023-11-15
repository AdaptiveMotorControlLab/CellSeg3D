"""This folder contains the code used by models in the plugin.

* ``models`` folder: contains the model classes, which are wrappers for the actual models. The wrappers are used to ensure that the models are compatible with the plugin.
* model_framework.py: contains the code for the model framework, used by training and inference plugins
* worker_inference.py: contains the code for the inference worker
* worker_training.py: contains the code for the training worker
* instance_segmentation.py: contains the code for instance segmentation
* crf.py: contains the code for the CRF postprocessing
* worker_utils.py: contains functions used by the workers

"""
