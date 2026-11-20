import numpy as np
import pytest
from napari._qt.qt_main_window import Window
from qtpy.QtWidgets import QPushButton

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
    assert len(widget.session.docks) == 2  # plots (bottom) + controls (left)
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


def test_controls_share_one_dock(review_setup):
    _, widget, _ = review_setup
    widget.run_review()

    controls = widget.session.docks[1].widget()
    assert controls.findChild(rev.Datamanager) is not None  # checked / timer
    labels = {b.text() for b in controls.findChildren(QPushButton)}
    assert {"Save", "Exit review", "Choose directory", "Open directory"} <= labels
    plots = widget.session.docks[0].widget()
    assert isinstance(plots, rev.FigureCanvas)


def test_finish_is_idempotent(review_setup):
    viewer, widget, _ = review_setup
    assert widget.finish_review(ask=False)  # no session: nothing to do

    widget.run_review()
    assert widget.finish_review(ask=False)
    assert widget.finish_review(ask=False)

    assert widget.session is None
    assert viewer.layers["other"].visible


def test_finish_survives_missing_dock_and_layer(review_setup):
    viewer, widget, _ = review_setup
    widget.run_review()
    viewer.window.remove_dock_widget(widget.session.docks[0])
    viewer.layers.remove("other")

    assert widget.finish_review(ask=False)

    assert widget.session is None
    assert widget.docked_widgets == []
    assert viewer.layers.selection.active is None or (
        viewer.layers.selection.active.name != "other"
    )


def test_failed_setup_restores_viewer(review_setup, monkeypatch):
    viewer, widget, _ = review_setup
    callbacks_before = list(viewer.mouse_drag_callbacks)
    added_docks, removed_docks = [], []
    original_add = Window.add_dock_widget
    original_remove = Window.remove_dock_widget

    def spy_add(self, *args, **kwargs):
        dock = original_add(self, *args, **kwargs)
        added_docks.append(dock)
        return dock

    def spy_remove(self, dock, *args, **kwargs):
        removed_docks.append(dock)
        return original_remove(self, dock, *args, **kwargs)

    def broken_prepare(*args, **kwargs):
        # by now the layers are hidden/locked, the plot dock and mouse callback installed
        raise ValueError("malformed csv")

    monkeypatch.setattr(Window, "add_dock_widget", spy_add)
    monkeypatch.setattr(Window, "remove_dock_widget", spy_remove)
    monkeypatch.setattr(rev.Datamanager, "prepare", broken_prepare)
    widget.run_review()

    assert widget.session is None
    assert viewer.layers["other"].visible
    assert not viewer.layers["image"].locked
    assert not viewer.layers["empty labels"].locked
    assert list(viewer.mouse_drag_callbacks) == callbacks_before
    assert added_docks
    assert removed_docks == added_docks
    assert widget.btn_start.text() == "Start reviewing"


def test_restart_does_not_stack_widgets(review_setup):
    viewer, widget, _ = review_setup
    callbacks_before = len(viewer.mouse_drag_callbacks)

    widget.run_review()
    widget.run_review()  # nothing modified: finishes the first session silently

    assert len(widget.session.docks) == 2  # plots (bottom) + controls (left)
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
    widget.session.save_dir.setText(str(out))

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


def test_finish_restores_selection_when_layers_were_added(review_setup):
    viewer, widget, _ = review_setup
    viewer.layers.remove("empty labels")  # the review adds a new labels layer
    viewer.layers.selection.active = viewer.layers["other"]

    widget.run_review()
    assert widget.finish_review(ask=False)

    assert viewer.layers.selection.active.name == "other"
    assert [layer.name for layer in viewer.layers.selection] == ["other"]


def _datamanager(widget):
    return widget.session.docks[1].widget().findChild(rev.Datamanager)


def test_slice_ignores_extra_viewer_axes(review_setup):
    viewer, widget, _ = review_setup
    # like a WNet3D prediction: (classes, Z, Y, X), hidden but still adds a viewer axis
    viewer.add_image(np.zeros((2, 8, 32, 32)), name="pred", visible=False)
    assert viewer.dims.ndim == 4
    viewer.dims.set_point(0, 1)  # class axis
    viewer.dims.set_point(1, 3)  # Z of the reviewed volume

    widget.run_review()
    dmg = _datamanager(widget)
    assert dmg.slice_num == 3

    viewer.dims.set_point(1, 6)
    assert dmg.slice_num == 6
    viewer.dims.set_point(0, 0)  # changing class must not move the reviewed slice
    assert dmg.slice_num == 6


def test_slice_follows_layer_scale(review_setup):
    viewer, widget, _ = review_setup
    viewer.layers["image"].scale = (2, 1, 1)
    viewer.layers["empty labels"].scale = (2, 1, 1)
    # an unscaled layer makes the viewer's Z step 1, so step index != data index
    viewer.add_image(np.zeros((16, 32, 32)), name="unscaled", visible=False)

    widget.run_review()
    viewer.dims.set_point(0, 10)  # world Z 10 is data slice 5

    assert _datamanager(widget).slice_num == 5


def test_review_without_labels_adds_empty_layer(review_setup):
    viewer, widget, _ = review_setup
    viewer.layers.remove("empty labels")
    n_layers = len(viewer.layers)

    widget.run_review()

    assert len(viewer.layers) == n_layers + 1
    assert widget.session.labels.data.shape == viewer.layers["image"].data.shape
    assert not widget.session.labels.data.any()


def test_empty_labels_follow_image_transform(review_setup):
    viewer, widget, _ = review_setup
    viewer.layers.remove("empty labels")
    image = viewer.layers["image"]
    image.scale = (2, 1, 1)
    image.translate = (0, 5, -3)
    image.rotate = 30
    assert not widget.anisotropy_widgets.enabled()

    widget.run_review()

    labels = widget.session.labels
    np.testing.assert_allclose(labels.scale, image.scale)
    np.testing.assert_allclose(labels.translate, image.translate)
    np.testing.assert_allclose(labels.rotate, image.rotate)
    np.testing.assert_allclose(labels.shear, image.shear)
    np.testing.assert_allclose(labels.affine.affine_matrix, image.affine.affine_matrix)


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
