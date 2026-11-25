# Inference

```python
from tifffile import imread

from napari_cellseg3d.config import ModelInfo
from napari_cellseg3d.dev_scripts import remote_inference as cs3d

image = imread("examples/c5image.tif")

config = cs3d.CONFIG  # InferenceWorkerConfig with defaults for a single volume
config.model_info = ModelInfo(name="SwinUNetR", model_input_size=[64, 64, 64], num_classes=2)

results = cs3d.inference_on_images(image, config=config)

labels, stats = cs3d.post_processing(
    results[0].semantic_segmentation,
    config=cs3d.PostProcessConfig(),
)
```

For WNet3D, pass `results[0].semantic_segmentation[1]` to `post_processing`.

## Functions

::: napari_cellseg3d.dev_scripts.remote_inference.inference_on_images

::: napari_cellseg3d.dev_scripts.remote_inference.post_processing

::: napari_cellseg3d.dev_scripts.remote_inference.PostProcessConfig

## Configuration and worker

- \[`InferenceWorkerConfig`\]\[napari_cellseg3d.config.InferenceWorkerConfig\]: device, model, weights, sliding window and post-processing settings.
- \[`ModelInfo`\]\[napari_cellseg3d.config.ModelInfo\]: model name, input size and number of classes.
- \[`SlidingWindowConfig`\]\[napari_cellseg3d.config.SlidingWindowConfig\]: window size and overlap.
- \[`InferenceResult`\]\[napari_cellseg3d.code_models.workers_utils.InferenceResult\]: what `inference_on_images` returns, one per image.
- \[`InferenceWorker`\]\[napari_cellseg3d.code_models.worker_inference.InferenceWorker\]: the worker `inference_on_images` runs.
