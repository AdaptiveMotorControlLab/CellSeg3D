import numpy as np
import pytest

from napari_cellseg3d import interface as ui
from napari_cellseg3d.code_plugins import plugin_review as rev


@pytest.fixture
def review_setup(make_napari_viewer_proxy, tmp_path):
    """A viewer with an image, an empty labels layer and an unrelated layer."""
    # Reviewer is a singleton: start each test with a fresh instance
    ui.QWidgetSingleton._instances.pop(rev.Reviewer, None)
    viewer = make_napari_viewer_proxy()
    image = np.random.default_rng(0).random((8, 32, 32)).astype(np.float32)
    viewer.add_image(image, name="image")
    viewer.add_labels(np.zeros(image.shape, dtype=np.uint16), name="empty labels")
    viewer.add_image(np.ones((32, 32)), name="other", visible=True)

    widget = rev.Reviewer(viewer)
    widget.layer_choice.setChecked(True)  # "New review": review existing layers
    widget.image_layer_loader.layer_list.setCurrentText("image")
    widget.label_layer_loader.layer_list.setCurrentText("empty labels")
    widget.results_filewidget.text_field.setText(str(tmp_path))
    widget.csv_textbox.setText("test")

    yield viewer, widget, tmp_path

    widget.finish_review(ask=False)
    ui.QWidgetSingleton._instances.pop(rev.Reviewer, None)


def test_review_stays_in_viewer(review_setup):
    viewer, widget, _ = review_setup
    n_layers = len(viewer.layers)

    widget.run_review()

    assert widget.session is not None
    assert widget._viewer == viewer
    assert len(viewer.layers) == n_layers  # reviews the empty labels in place
    assert widget.session.labels.name == "empty labels"
    assert not viewer.layers["other"].visible
    assert viewer.layers.selection.active.name == "empty labels"
    assert len(widget.session.docks) == 3
    assert widget.btn_start.text() == "Finish review"


def test_finish_restores_viewer(review_setup):
    viewer, widget, _ = review_setup
    callbacks_before = list(viewer.mouse_drag_callbacks)

    widget.run_review()
    assert widget.finish_review(ask=False)

    assert widget.session is None
    assert viewer.layers["other"].visible
    assert list(viewer.mouse_drag_callbacks) == callbacks_before
    assert widget.docked_widgets == []
    assert widget.btn_start.text() == "Start reviewing"
    assert not viewer.layers["empty labels"].locked


def test_restart_does_not_stack_widgets(review_setup):
    viewer, widget, _ = review_setup
    callbacks_before = len(viewer.mouse_drag_callbacks)

    widget.run_review()
    widget.run_review()  # nothing modified: finishes the first session silently

    assert len(widget.session.docks) == 3
    assert len(viewer.mouse_drag_callbacks) == callbacks_before + 1


def test_save_labels(review_setup):
    viewer, widget, tmp_path = review_setup
    widget.run_review()
    widget.session.labels.data[0, 0, 0] = 3
    widget.session.modified = True

    path = widget.save_labels(tmp_path)

    assert path.is_file()
    assert not widget.session.modified


def test_save_button_writes_to_chosen_folder(review_setup, tmp_path):
    viewer, widget, _ = review_setup
    widget.run_review()
    out = tmp_path / "chosen"
    out.mkdir()
    widget.session.save_dir.value = out

    widget.save_labels()

    assert (out / "labels_reviewed.tif").is_file()


def test_open_directory(monkeypatch, tmp_path):
    opened = []
    monkeypatch.setattr(
        ui.QDesktopServices, "openUrl", lambda url: opened.append(url) or True
    )

    assert ui.open_directory(tmp_path)
    assert opened[0].toLocalFile().replace("\\", "/") == str(
        tmp_path.resolve()
    ).replace("\\", "/")
    assert not ui.open_directory(tmp_path / "missing")
    assert len(opened) == 1


def test_review_without_labels_adds_empty_layer(review_setup):
    viewer, widget, _ = review_setup
    viewer.layers.remove("empty labels")
    n_layers = len(viewer.layers)

    widget.run_review()

    assert len(viewer.layers) == n_layers + 1
    assert widget.session.labels.data.shape == viewer.layers["image"].data.shape
    assert not widget.session.labels.data.any()


def test_anisotropy_unchecked_keeps_layer_scale(review_setup):
    viewer, widget, _ = review_setup
    viewer.layers["image"].scale = (2, 1, 1)
    assert not widget.anisotropy_widgets.enabled()

    widget.run_review()

    assert widget.config.zoom_factor == [1, 1, 1]
    assert tuple(viewer.layers["image"].scale) == (2, 1, 1)


def test_anisotropy_checked_sets_layer_scale(review_setup):
    viewer, widget, _ = review_setup
    widget.anisotropy_widgets.checkbox.setChecked(True)

    widget.run_review()

    expected = tuple(widget.anisotropy_widgets.scaling_zyx())
    assert tuple(viewer.layers["image"].scale) == pytest.approx(expected)
    assert tuple(viewer.layers["empty labels"].scale) == pytest.approx(expected)


def test_projection_style_follows_layer(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    layer = viewer.add_image(
        np.zeros((4, 4, 4), np.float32), colormap="viridis", contrast_limits=(10, 50)
    )
    style = rev.layer_mpl_style(layer)
    assert (style["norm"].vmin, style["norm"].vmax) == (10, 50)
    assert np.allclose(style["cmap"](1.0), layer.colormap.map([1.0])[0])

    layer.gamma = 0.5
    assert rev.layer_mpl_style(layer)["norm"].gamma == 0.5


def test_finish_disconnects_projection_redraw(review_setup):
    viewer, widget, _ = review_setup
    events = viewer.layers["image"].events.contrast_limits
    n_callbacks = len(events.callbacks)

    widget.run_review()
    assert len(events.callbacks) == n_callbacks + 1
    widget.finish_review(ask=False)

    assert len(events.callbacks) == n_callbacks


def test_crop_keeps_float_values():
    data = np.full((20, 20, 20), 0.25, dtype=np.float32)
    crop = rev.crop_volume_around_point([10, 10, 10], data, [1, 1, 1])
    assert crop.dtype == np.float32
    assert crop[50, 50, 50] == 0.25


def test_crop_volume_around_point_pads_edges():
    data = np.ones((20, 20, 20), dtype=np.int16)
    crop = rev.crop_volume_around_point([0, 0, 0], data, [1, 1, 1])
    assert crop.shape == (100, 100, 100)
    assert crop[50:70, 50:70, 50:70].all()
    assert not crop[:50].any()
