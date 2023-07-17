from pathlib import Path

from napari_cellseg3d.code_plugins.plugin_base import (
    BasePluginSingleImage,
)


def test_base_single_image(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    plugin = BasePluginSingleImage(viewer)

    test_folder = Path(__file__).parent.resolve()
    test_image = str(test_folder / "res/test.tif")

    assert plugin._check_results_path(str(test_folder))
    plugin.image_path = test_image
    assert plugin._default_path[0] != test_image
    plugin._update_default_paths()
    assert plugin._default_path[0] == test_image
