from pathlib import Path
from types import SimpleNamespace

import pytest
from qtpy.QtWidgets import QWidget

from napari_cellseg3d.code_plugins import plugin_base
from napari_cellseg3d.code_plugins.plugin_base import (
    BasePluginFolder,
    BasePluginSingleImage,
)


def test_base_single_image_update_default_paths(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    plugin = BasePluginSingleImage(viewer)

    test_folder = Path(__file__).parent.resolve()
    test_image = str(test_folder / "res/test.tif")

    assert plugin._check_results_path(str(test_folder))

    plugin.image_path = test_image
    assert plugin._default_path[0] != test_image

    plugin._update_default_paths()

    assert plugin._default_path == [test_image, None, None]


def test_check_results_path_creates_missing_folder(
    make_napari_viewer_proxy, tmp_path
):
    plugin = BasePluginSingleImage(make_napari_viewer_proxy())

    results_dir = tmp_path / "new" / "results"

    assert not results_dir.exists()
    assert plugin._check_results_path(str(results_dir))
    assert results_dir.is_dir()


def test_check_results_path_empty_string_returns_false(
    make_napari_viewer_proxy,
):
    plugin = BasePluginSingleImage(make_napari_viewer_proxy())

    assert plugin._check_results_path("") is False


def test_check_results_path_rejects_non_string(make_napari_viewer_proxy):
    plugin = BasePluginSingleImage(make_napari_viewer_proxy())

    with pytest.raises(TypeError, match="Expected string"):
        plugin._check_results_path(None)


def test_single_image_build_not_implemented(make_napari_viewer_proxy):
    plugin = BasePluginSingleImage(make_napari_viewer_proxy())

    with pytest.raises(
        NotImplementedError, match="To be defined in child classes"
    ):
        plugin._build()


def test_make_navigation_buttons(make_napari_viewer_proxy):
    plugin = BasePluginSingleImage(make_napari_viewer_proxy())

    plugin.addTab(SimpleNamespace(), "A")
    plugin.addTab(SimpleNamespace(), "B")
    plugin.setCurrentIndex(1)

    prev_button = plugin._make_prev_button()
    next_button = plugin._make_next_button()

    prev_button.click()
    assert plugin.currentIndex() == 0

    next_button.click()
    assert plugin.currentIndex() == 1


def test_remove_docked_widgets_success(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    plugin = BasePluginSingleImage(viewer)

    dock = viewer.window.add_dock_widget(plugin, name="temporary dock")
    plugin.docked_widgets = [dock]
    plugin.container_docked = True

    assert plugin.remove_docked_widgets() is True
    assert plugin.docked_widgets == []
    assert plugin.container_docked is False


def test_remove_docked_widgets_handles_lookup_error(
    make_napari_viewer_proxy, monkeypatch
):
    viewer = make_napari_viewer_proxy()
    plugin = BasePluginSingleImage(viewer)

    plugin.docked_widgets = [object()]
    plugin.container_docked = True

    def raise_lookup_error(_dock_widget):
        raise LookupError

    monkeypatch.setattr(
        viewer.window, "remove_dock_widget", raise_lookup_error
    )

    assert plugin.remove_docked_widgets() is False


def test_extract_dataset_paths_empty():
    assert BasePluginFolder.extract_dataset_paths([]) is None


def test_extract_dataset_paths_none():
    assert BasePluginFolder.extract_dataset_paths([None]) is None


def test_extract_dataset_paths_returns_parent(tmp_path):
    image_path = tmp_path / "images" / "image.tif"
    image_path.parent.mkdir()
    image_path.write_text("fake")

    assert BasePluginFolder.extract_dataset_paths([str(image_path)]) == str(
        image_path.parent
    )


def test_folder_update_default_paths_from_existing_paths(
    make_napari_viewer_proxy, tmp_path
):
    plugin = BasePluginFolder(make_napari_viewer_proxy())

    image_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    val_dir = tmp_path / "validation"
    results_dir = tmp_path / "results"

    for folder in [image_dir, label_dir, val_dir, results_dir]:
        folder.mkdir()

    plugin.images_filepaths = [str(image_dir / "img.tif")]
    plugin.labels_filepaths = [str(label_dir / "lab.tif")]
    plugin.validation_filepaths = [str(val_dir / "val.tif")]
    plugin.results_path = str(results_dir)

    plugin._update_default_paths()

    assert plugin._default_path == [
        str(image_dir),
        str(label_dir),
        str(val_dir),
        str(results_dir),
    ]


def test_folder_update_default_paths_appends_existing_dir(
    make_napari_viewer_proxy,
    tmp_path,
):
    plugin = BasePluginFolder(make_napari_viewer_proxy())

    plugin._update_default_paths(str(tmp_path))

    assert str(tmp_path) in plugin._default_path


def test_load_dataset_paths(make_napari_viewer_proxy, monkeypatch, tmp_path):
    plugin = BasePluginFolder(make_napari_viewer_proxy())

    image_0 = tmp_path / "0.tif"
    image_1 = tmp_path / "1.tif"
    image_0.write_text("fake")
    image_1.write_text("fake")

    expected = [image_0, image_1]

    monkeypatch.setattr(
        plugin_base.ui,
        "open_folder_dialog",
        lambda *_args, **_kwargs: str(tmp_path),
    )
    monkeypatch.setattr(
        plugin_base.utils,
        "get_all_matching_files",
        lambda _directory: expected,
    )

    assert plugin.load_dataset_paths() == expected


def test_load_dataset_paths_warns_when_empty(
    make_napari_viewer_proxy,
    monkeypatch,
    tmp_path,
):
    plugin = BasePluginFolder(make_napari_viewer_proxy())

    warnings = []

    monkeypatch.setattr(
        plugin_base.ui,
        "open_folder_dialog",
        lambda *_args, **_kwargs: str(tmp_path),
    )
    monkeypatch.setattr(
        plugin_base.utils,
        "get_all_matching_files",
        lambda _directory: [],
    )
    monkeypatch.setattr(
        plugin_base.logger,
        "warning",
        lambda msg: warnings.append(msg),
    )

    assert plugin.load_dataset_paths() == []
    assert warnings
    assert "does not contain any compatible" in warnings[0]


def test_load_image_dataset(make_napari_viewer_proxy, monkeypatch, tmp_path):
    plugin = BasePluginFolder(make_napari_viewer_proxy())

    image_0 = tmp_path / "b.tif"
    image_1 = tmp_path / "a.tif"
    image_0.write_text("fake")
    image_1.write_text("fake")

    monkeypatch.setattr(
        plugin, "load_dataset_paths", lambda: [image_0, image_1]
    )

    plugin.load_image_dataset()

    assert plugin.images_filepaths == [str(image_1), str(image_0)]
    assert plugin.image_filewidget.text_field.text() == str(tmp_path)


def test_load_label_dataset(make_napari_viewer_proxy, monkeypatch, tmp_path):
    plugin = BasePluginFolder(make_napari_viewer_proxy())

    label_0 = tmp_path / "b.tif"
    label_1 = tmp_path / "a.tif"
    label_0.write_text("fake")
    label_1.write_text("fake")

    monkeypatch.setattr(
        plugin, "load_dataset_paths", lambda: [label_0, label_1]
    )

    plugin.load_label_dataset()

    assert plugin.labels_filepaths == [str(label_1), str(label_0)]
    assert plugin.labels_filewidget.text_field.text() == str(tmp_path)


def test_load_unsup_images_dataset(
    make_napari_viewer_proxy, monkeypatch, tmp_path
):
    plugin = BasePluginFolder(make_napari_viewer_proxy())

    image_0 = tmp_path / "b.tif"
    image_1 = tmp_path / "a.tif"
    image_0.write_text("fake")
    image_1.write_text("fake")

    monkeypatch.setattr(
        plugin, "load_dataset_paths", lambda: [image_0, image_1]
    )

    plugin.load_unsup_images_dataset()

    assert plugin.validation_filepaths == [str(image_1), str(image_0)]
    assert plugin.unsupervised_images_filewidget.text_field.text() == str(
        tmp_path
    )


def test_show_file_dialog_updates_filetype(
    make_napari_viewer_proxy,
    monkeypatch,
    tmp_path,
):
    plugin = BasePluginSingleImage(make_napari_viewer_proxy())

    image_path = tmp_path / "image.tif"
    image_path.write_text("fake")

    monkeypatch.setattr(
        plugin_base.ui,
        "open_file_dialog",
        lambda *_args, **_kwargs: [str(image_path)],
    )

    result = plugin._show_file_dialog()

    assert result == str(image_path)
    assert plugin.filetype == ".tif"


def test_show_dialog_images_sets_image_path(
    make_napari_viewer_proxy,
    monkeypatch,
    tmp_path,
):
    plugin = BasePluginSingleImage(make_napari_viewer_proxy())

    image_path = tmp_path / "image.tif"
    image_path.write_text("fake")

    monkeypatch.setattr(plugin, "_show_file_dialog", lambda: str(image_path))

    plugin._show_dialog_images()

    assert plugin.image_path == str(image_path)
    assert plugin.image_filewidget.text_field.text() == str(image_path)


def test_show_dialog_labels_sets_label_path(
    make_napari_viewer_proxy,
    monkeypatch,
    tmp_path,
):
    plugin = BasePluginSingleImage(make_napari_viewer_proxy())

    label_path = tmp_path / "label.tif"
    label_path.write_text("fake")

    monkeypatch.setattr(plugin, "_show_file_dialog", lambda: str(label_path))

    plugin._show_dialog_labels()

    assert plugin.label_path == str(label_path)
    assert plugin.labels_filewidget.text_field.text() == str(label_path)


def test_load_results_path_sets_results_path(
    make_napari_viewer_proxy,
    monkeypatch,
    tmp_path,
):
    plugin = BasePluginSingleImage(make_napari_viewer_proxy())

    monkeypatch.setattr(
        plugin_base.ui,
        "open_folder_dialog",
        lambda *_args, **_kwargs: str(tmp_path),
    )

    plugin._load_results_path()

    assert plugin.results_path == str(tmp_path.resolve())
    assert plugin.results_filewidget.text_field.text() == str(
        tmp_path.resolve()
    )


def test_show_and_hide_io_element_without_toggle(qtbot):
    widget = QWidget()
    widget.setVisible(False)
    qtbot.addWidget(widget)

    BasePluginSingleImage._show_io_element(widget)

    assert widget.isVisible()

    BasePluginSingleImage._hide_io_element(widget)

    assert not widget.isVisible()
