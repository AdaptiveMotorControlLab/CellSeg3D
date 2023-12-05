import random
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import torch

from napari_cellseg3d import utils
from napari_cellseg3d.dev_scripts import thread_test

rand_gen = utils.rand_gen


def test_singleton_class():
    class TestSingleton(metaclass=utils.Singleton):
        def __init__(self, value):
            self.value = value

    a = TestSingleton(1)
    b = TestSingleton(2)

    assert a.value == b.value


def test_save_folder():
    test_path = Path(__file__).resolve().parent / "res"
    folder_name = "test_folder"
    images = [rand_gen.random((5, 5, 5)).astype(np.float32) for _ in range(10)]
    images_paths = [f"{i}.tif" for i in range(10)]

    utils.save_folder(
        test_path, folder_name, images, images_paths, exist_ok=True
    )
    assert (test_path / folder_name).is_dir()
    for i in range(10):
        assert (test_path / folder_name / images_paths[i]).is_file()


def test_normalize_y():
    test_array = np.array([0, 255, 127.5])
    results = utils.normalize_y(test_array)
    expected = test_array / 255
    assert np.all(results == expected)
    assert np.all(test_array == utils.denormalize_y(results))


def test_sphericities():
    for _i in range(100):
        mock_volume = random.randint(1, 10)
        mock_surface = random.randint(
            100, 1000
        )  # assuming surface is always larger than volume
        sphericity_vol = utils.sphericity_volume_area(
            mock_volume, mock_surface
        )
        assert 0 <= sphericity_vol <= 1

        semi_major = random.randint(10, 100)
        semi_minor = random.randint(10, 100)
        try:
            sphericity_axes = utils.sphericity_axis(semi_major, semi_minor)
        except ZeroDivisionError:
            sphericity_axes = 0
        except ValueError:
            sphericity_axes = 0
        if sphericity_axes is None:
            sphericity_axes = (
                0  # errors already handled in function, returns None
            )
        assert 0 <= sphericity_axes <= 1


def test_normalize_max():
    test_array = np.array([0, 255, 127.5])
    expected = np.array([0, 1, 0.5])
    assert np.all(utils.normalize_max(test_array) == expected)


def test_dice_coeff():
    test_array = rand_gen.integers(0, 2, (64, 64, 64))
    test_array_2 = rand_gen.integers(0, 2, (64, 64, 64))
    assert utils.dice_coeff(test_array, test_array) == 1
    assert utils.dice_coeff(test_array, test_array_2) <= 1


def test_fill_list_in_between():
    test_list = [1, 2, 3, 4, 5, 6]
    res = [
        1,
        "",
        "",
        2,
        "",
        "",
        3,
        "",
        "",
        4,
        "",
        "",
        5,
        "",
        "",
        6,
        "",
        "",
    ]

    assert utils.fill_list_in_between(test_list, 2, "") == res

    fill = partial(utils.fill_list_in_between, n=2, fill_value="")

    assert fill(test_list) == res


def test_align_array_sizes():
    im = np.zeros((128, 512, 256))
    print(im.shape)

    dim_1 = (64, 64, 512)
    ground = np.array((512, 64, 64))
    pred = np.array(dim_1)

    ori, targ = utils.align_array_sizes(ground, pred)

    im_1 = np.moveaxis(im, ori, targ)
    print(im_1.shape)
    assert im_1.shape == (512, 256, 128)

    dim_2 = (512, 256, 128)
    ground = np.array((128, 512, 256))
    pred = np.array(dim_2)

    ori, targ = utils.align_array_sizes(ground, pred)

    im_2 = np.moveaxis(im, ori, targ)
    print(im_2.shape)
    assert im_2.shape == dim_2

    dim_3 = (128, 128, 128)
    ground = np.array(dim_3)
    pred = np.array(dim_3)

    ori, targ = utils.align_array_sizes(ground, pred)
    im_3 = np.moveaxis(im, ori, targ)
    print(im_3.shape)
    assert im_3.shape == im.shape


def test_get_padding_dim():
    tensor = torch.randn(100, 30, 40)
    size = tensor.size()

    pad = utils.get_padding_dim(size)

    assert pad == [128, 32, 64]

    tensor = torch.randn(2000, 30, 40)
    size = tensor.size()

    # warn = logger.warning(
    #     "Warning : a very large dimension for automatic padding has been computed.\n"
    #     "Ensure your images are of an appropriate size and/or that you have enough memory."
    #     "The padding value is currently 2048."
    # )
    #
    pad = utils.get_padding_dim(size)
    #
    # pytest.warns(warn, (lambda: utils.get_padding_dim(size)))

    assert pad == [2048, 32, 64]

    tensor = torch.randn(65, 70, 80)
    size = tensor.size()

    pad = utils.get_padding_dim(size)

    assert pad == [128, 128, 128]

    tensor_wrong = torch.randn(65, 70, 80, 90)
    with pytest.raises(
        ValueError,
        match="Please check the dimensions of the input, only 2 or 3-dimensional data is supported currently",
    ):
        utils.get_padding_dim(tensor_wrong.size())


def test_normalize_x():
    test_array = utils.normalize_x(np.array([0, 255, 127.5]))
    expected = np.array([-1, 1, 0])
    assert np.all(test_array == expected)


def test_load_images():
    path = Path(__file__).resolve().parent / "res"
    # with pytest.raises(
    #     ValueError, match="If loading as a folder, filetype must be specified"
    # ):
    #     images = utils.load_images(str(path), as_folder=True)
    # with pytest.raises(
    #     NotImplementedError,
    #     match="Loading as folder not implemented yet. Use napari to load as folder",
    # ):
    #     images = utils.load_images(str(path), as_folder=True, filetype=".tif")
    # # assert len(images) == 1
    path = path / "test.tif"
    images = utils.load_images(str(path))
    assert images.shape == (6, 6, 6)


def test_parse_default_path():
    user_path = Path.home()
    assert utils.parse_default_path([None]) == str(user_path)

    test_path = (Path.home() / "test" / "test" / "test" / "test").as_posix()
    path = [test_path, None, None]
    assert utils.parse_default_path(path, check_existence=False) == str(
        test_path
    )

    test_path = (Path.home() / "test" / "does" / "not" / "exist").as_posix()
    path = [test_path, None, None]
    assert utils.parse_default_path(path, check_existence=True) == str(
        Path.home()
    )

    long_path = Path.home()
    long_path = (
        long_path
        / "very"
        / "long"
        / "path"
        / "what"
        / "a"
        / "bore"
        / "ifonlytherewassomething"
        / "tohelpmenotsearchit"
        / "allthetime"
    )
    path = [test_path, None, None, long_path, ""]
    assert utils.parse_default_path(path, check_existence=False) == str(
        long_path.as_posix()
    )


def test_thread_test(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    w = thread_test.create_connected_widget(viewer)
    viewer.window.add_dock_widget(w)


def test_quantile_norm():
    array = rand_gen.random(size=(100, 100, 100))
    low_quantile = np.quantile(array, 0.01)
    high_quantile = np.quantile(array, 0.99)
    array_norm = utils.quantile_normalization(array)
    assert array_norm.min() >= low_quantile
    assert array_norm.max() <= high_quantile


def test_get_all_matching_files():
    test_image_path = Path(__file__).resolve().parent / "res/wnet_test"
    paths = utils.get_all_matching_files(test_image_path)

    assert len(paths) == 1
    assert [Path(p).is_file() for p in paths]
    assert [Path(p).suffix == ".tif" for p in paths]
