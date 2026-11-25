# Training

## WNet3D (self-supervised)

```python
from pathlib import Path

from napari_cellseg3d.config import WandBConfig, WeightsInfo, WNetTrainingWorkerConfig
from napari_cellseg3d.dev_scripts import colab_training as c

config = WNetTrainingWorkerConfig(
    device="cuda:0",
    max_epochs=50,
    learning_rate=2e-5,
    validation_interval=2,
    batch_size=4,
    num_workers=2,
    weights_info=WeightsInfo(),
    results_path_folder="results",
    train_data_dict=c.create_dataset_dict_no_labs(Path("train_images")),
    eval_volume_dict=c.create_eval_dataset_dict(Path("eval_images"), Path("eval_labels")),  # or None
)

Path(config.results_path_folder).mkdir(parents=True, exist_ok=True)
worker = c.get_colab_worker(worker_config=config, wandb_config=WandBConfig(mode="disabled"))
for _report in worker.train():
    pass
```

::: napari_cellseg3d.dev_scripts.colab_training.get_colab_worker

::: napari_cellseg3d.dev_scripts.colab_training.create_dataset_dict_no_labs

::: napari_cellseg3d.dev_scripts.colab_training.create_eval_dataset_dict

- \[`WNetTrainingWorkerConfig`\]\[napari_cellseg3d.config.WNetTrainingWorkerConfig\]: WNet3D hyperparameters and data.
- \[`WNetTrainingWorker`\]\[napari_cellseg3d.code_models.worker_training.WNetTrainingWorker\]: `train()` is a generator yielding a `TrainingReport` per epoch and per validation.

## Supervised models

Supervised models (SegResNet, SwinUNetR, …) use the same pattern with their own config and worker:

- \[`SupervisedTrainingWorkerConfig`\][napari_cellseg3d.config.SupervisedTrainingWorkerConfig]
- \[`SupervisedTrainingWorker`\][napari_cellseg3d.code_models.worker_training.SupervisedTrainingWorker]

## Shared configuration

- \[`TrainingWorkerConfig`\]\[napari_cellseg3d.config.TrainingWorkerConfig\]: fields common to all training configs.
- \[`WeightsInfo`\]\[napari_cellseg3d.config.WeightsInfo\]: pretrained or custom weights.
- \[`WandBConfig`\]\[napari_cellseg3d.config.WandBConfig\]: Weights & Biases logging.
