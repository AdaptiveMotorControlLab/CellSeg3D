from pathlib import Path
from napari_cellseg3d.code_models.models.wnet import train_wnet as t

def test_wnet_training():
    config = t.Config()

    config.batch_size = 1
    config.num_epochs = 1

    config.train_volume_directory = str(Path(__file__).resolve().parent / "res/wnet_test")
    config.eval_volume_directory = config.train_volume_directory
    config.save_every = 1
    config.val_interval = 2  # skip validation
    config.save_model_path = config.train_volume_directory + "/test.pth"

    ncuts_loss, rec_loss, model = t.train(train_config=config)

    assert ncuts_loss is not None
    assert rec_loss is not None
    assert model is not None
