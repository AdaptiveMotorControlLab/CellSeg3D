"""Showcases how to train a model without napari."""

from pathlib import Path

from napari_cellseg3d import config as cfg
from napari_cellseg3d.code_models.worker_training import (
    SupervisedTrainingWorker,
)
from napari_cellseg3d.utils import LOGGER as logger
from napari_cellseg3d.utils import get_all_matching_files

TRAINING_SPLIT = 0.2  # 0.4, 0.8
MODEL_NAME = "SegResNet"  # "SwinUNetR"
BATCH_SIZE = 10 if MODEL_NAME == "SegResNet" else 5
# BATCH_SIZE = 1

SPLIT_FOLDER = "1_c15"  # "2_c1_c4_visual"  "3_c1245_visual"
RESULTS_PATH = (
    Path("/data/cyril")
    / "CELLSEG_BENCHMARK/cellseg3d_train"
    / f"{MODEL_NAME}_{SPLIT_FOLDER}_{int(TRAINING_SPLIT*100)}"
)

IMAGES = (
    Path("/data/cyril")
    / f"CELLSEG_BENCHMARK/TPH2_mesospim/SPLITS/{SPLIT_FOLDER}"
)
LABELS = (
    Path("/data/cyril")
    / f"CELLSEG_BENCHMARK/TPH2_mesospim/SPLITS/{SPLIT_FOLDER}/labels/semantic"
)


class LogFixture:
    """Fixture for napari-less logging, replaces napari_cellseg3d.interface.Log in model_workers.

    This allows to redirect the output of the workers to stdout instead of a specialized widget.
    """

    def __init__(self):
        """Creates a LogFixture object."""
        super(LogFixture, self).__init__()

    def print_and_log(self, text, printing=None):
        """Prints and logs text."""
        print(text)

    def warn(self, warning):
        """Logs warning."""
        logger.warning(warning)

    def error(self, e):
        """Logs error."""
        raise (e)


def prepare_data(images_path, labels_path):
    """Prepares data for training."""
    assert images_path.exists(), f"Images path does not exist: {images_path}"
    assert labels_path.exists(), f"Labels path does not exist: {labels_path}"
    if not RESULTS_PATH.exists():
        RESULTS_PATH.mkdir(parents=True, exist_ok=True)

    images = get_all_matching_files(images_path)
    labels = get_all_matching_files(labels_path)

    print(f"Images paths: {images}")
    print(f"Labels paths: {labels}")

    logger.info("Images :\n")
    for file in images:
        logger.info(Path(file).name)
    logger.info("*" * 10)
    logger.info("Labels :\n")
    for file in images:
        logger.info(Path(file).name)

    assert len(images) == len(
        labels
    ), "Number of images and labels must be the same"

    return [
        {"image": str(image_path), "label": str(label_path)}
        for image_path, label_path in zip(images, labels)
    ]


def remote_training():
    """Function to train a model without napari."""
    # print(f"Results path: {RESULTS_PATH.resolve()}")

    wandb_config = cfg.WandBConfig(
        mode="online",
        save_model_artifact=True,
    )

    deterministic_config = cfg.DeterministicConfig(
        seed=34936339,
    )

    worker_config = cfg.SupervisedTrainingWorkerConfig(
        device="cuda:0",
        max_epochs=50,
        learning_rate=0.001,  # 1e-3
        validation_interval=2,
        batch_size=BATCH_SIZE,  # 10 for SegResNet
        deterministic_config=deterministic_config,
        scheduler_factor=0.5,
        scheduler_patience=10,  # use default scheduler
        weights_info=cfg.WeightsInfo(),  # no pretrained weights
        results_path_folder=str(RESULTS_PATH),
        sampling=False,
        do_augmentation=True,
        train_data_dict=prepare_data(IMAGES, LABELS),
        # supervised specific
        model_info=cfg.ModelInfo(
            name=MODEL_NAME,
            model_input_size=(64, 64, 64),
        ),
        loss_function="Generalized Dice",
        training_percent=TRAINING_SPLIT,
    )

    worker = SupervisedTrainingWorker(worker_config)
    worker.wandb_config = wandb_config
    ######### SET LOG
    log = LogFixture()
    worker.log_signal.connect(log.print_and_log)
    worker.warn_signal.connect(log.warn)
    worker.error_signal.connect(log.error)

    results = []
    for result in worker.train():
        results.append(result)
    print("Training finished")


if __name__ == "__main__":
    results = remote_training()
