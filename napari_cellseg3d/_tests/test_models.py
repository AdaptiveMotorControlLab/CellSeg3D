from napari_cellseg3d.config import MODEL_LIST


def test_model_list():
    for model_name in MODEL_LIST.keys():
        dims = 128
        test = MODEL_LIST[model_name](
            input_img_size=[dims, dims, dims],
            in_channels=1,
            out_channels=1,
            dropout_prob=0.3,
        )
        assert isinstance(test, MODEL_LIST[model_name])
