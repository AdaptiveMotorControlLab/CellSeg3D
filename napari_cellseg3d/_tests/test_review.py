from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
from napari._qt.qt_main_window import Window
from qtpy.QtWidgets import QPushButton
from tifffile import imread

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


def test_projection_labels_match_data():
    # each voxel holds its own coordinates, so a section shows which axes it spans
    z, y, x = np.indices((100, 100, 100))
    crop = {"Z axis": z, "Y axis": y, "X axis": x}

    for plane, xlabel, ylabel in rev.PROJECTIONS:
        rows = rev.projection(crop[ylabel], plane)
        cols = rev.projection(crop[xlabel], plane)
        # the y-labelled axis varies down rows, the x-labelled one along columns
        np.testing.assert_array_equal(rows[:, 0], np.arange(100))
        np.testing.assert_array_equal(cols[0, :], np.arange(100))


def test_figure_uses_projection_labels(review_setup):
    _, widget, _ = review_setup
    widget.run_review()

    figure = widget.session.docks[0].widget().figure
    labels = [(a.get_xlabel(), a.get_ylabel()) for a in figure.axes]
    assert labels == [(xlabel, ylabel) for _, xlabel, ylabel in rev.PROJECTIONS]


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


class _ShiftClick:
    modifiers = ("shift",)


def _shift_click(viewer, widget, world_position):
    viewer.cursor.position = world_position
    widget.session.mouse_callback(viewer, _ShiftClick())
    figure = widget.session.docks[0].widget().figure
    return [np.asarray(axes.images[0].get_array()) for axes in figure.axes]


@pytest.mark.parametrize("anisotropic", [False, True])
def test_shift_click_plots_the_shown_slice(review_setup, anisotropic):
    viewer, widget, _ = review_setup
    image = viewer.layers["image"]
    image.data = np.broadcast_to(
        np.arange(8, dtype=np.float32)[:, None, None], (8, 32, 32)
    ).copy()  # each voxel holds its Z index
    widget.anisotropy_widgets.checkbox.setChecked(anisotropic)

    widget.run_review()
    scale_z = image.scale[0]
    for z in range(8):
        viewer.dims.set_point(0, z * scale_z)
        xy, _, _ = _shift_click(viewer, widget, (z * scale_z, 16, 16))
        assert _datamanager(widget).slice_num == z
        assert xy[50, 50] == z


def test_shift_click_ignores_extra_viewer_axes(review_setup):
    viewer, widget, _ = review_setup
    image = viewer.layers["image"]
    image.data = np.broadcast_to(
        np.arange(8, dtype=np.float32)[:, None, None], (8, 32, 32)
    ).copy()
    viewer.add_image(np.zeros((2, 8, 32, 32)), name="pred", visible=False)

    widget.run_review()
    xy, _, _ = _shift_click(viewer, widget, (1, 3, 16, 16))

    assert xy[50, 50] == 3


def test_crop_keeps_float_values():
    data = np.full((20, 20, 20), 0.25, dtype=np.float32)
    crop = rev.crop_volume_around_point([10, 10, 10], data)
    assert crop.dtype == np.float32
    assert crop[50, 50, 50] == 0.25


def test_crop_volume_around_point_pads_edges():
    data = np.ones((20, 20, 20), dtype=np.int16)
    crop = rev.crop_volume_around_point([0, 0, 0], data)
    assert crop.shape == (100, 100, 100)
    assert crop[50:70, 50:70, 50:70].all()
    assert not crop[:50].any()


def _answer_prompt(monkeypatch, answer):
    """Makes the save prompt return `answer`; returns the buttons it was offered."""
    offered = []

    def question(parent, title, text, buttons, default):
        offered.append(buttons)
        return answer

    monkeypatch.setattr(rev.QMessageBox, "question", question)
    return offered


def _paint(widget):
    widget.session.labels.paint((0, 5, 5), 7)
    assert widget.session.modified  # painting marks the session as modified


@pytest.mark.parametrize(
    ("answer", "saved", "finished"),
    [
        (rev.QMessageBox.Save, True, True),
        (rev.QMessageBox.Discard, False, True),
        (rev.QMessageBox.Cancel, False, False),
    ],
)
def test_finish_prompts_when_labels_painted(
    review_setup, monkeypatch, answer, saved, finished
):
    viewer, widget, tmp_path = review_setup
    widget.run_review()
    session = widget.session
    callbacks = list(viewer.mouse_drag_callbacks)
    _paint(widget)
    offered = _answer_prompt(monkeypatch, answer)

    assert widget.finish_review() is finished

    assert len(offered) == 1
    out = tmp_path / "labels_reviewed.tif"
    assert out.is_file() is saved
    if saved:
        assert imread(out)[0, 5, 5] == 7
    if finished:
        assert widget.session is None
        assert viewer.layers["other"].visible
    else:  # cancelled: the review goes on untouched
        assert widget.session is session
        assert session.modified
        assert not viewer.layers["other"].visible
        assert viewer.layers["empty labels"].locked
        assert list(viewer.mouse_drag_callbacks) == callbacks
        assert len(session.docks) == 2
        assert widget.btn_start.text() == "Finish review"


def test_close_with_painted_labels_can_be_cancelled(review_setup, monkeypatch):
    viewer, widget, _ = review_setup
    viewer.window.add_dock_widget(widget)
    widget.run_review()
    _paint(widget)
    _answer_prompt(monkeypatch, rev.QMessageBox.Cancel)

    widget.remove_from_viewer()

    assert widget.session is not None
    assert widget.parent() is not None  # still docked


def test_failed_save_keeps_session(review_setup, monkeypatch):
    viewer, widget, _ = review_setup
    widget.run_review()
    session = widget.session
    _paint(widget)
    _answer_prompt(monkeypatch, rev.QMessageBox.Save)

    def broken_imwrite(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(rev, "imwrite", broken_imwrite)
    with pytest.raises(OSError, match="disk full"):
        widget.finish_review()

    assert widget.session is session  # can retry
    assert session.modified
    assert not viewer.layers["other"].visible


def test_dock_closed_from_title_bar_finishes_review(review_setup, monkeypatch, qtbot):
    viewer, widget, _ = review_setup
    viewer.window.add_dock_widget(widget)
    widget.run_review()
    _paint(widget)
    offered = _answer_prompt(monkeypatch, rev.QMessageBox.Discard)

    # what the title-bar X ends up doing, without going through remove_from_viewer
    viewer.window.remove_dock_widget(widget)
    qtbot.waitUntil(lambda: widget.session is None)

    assert viewer.layers["other"].visible
    assert not viewer.layers["empty labels"].locked
    assert len(offered) == 1
    assert not (offered[0] & rev.QMessageBox.Cancel)  # the dock is already gone


def test_finish_restores_hidden_review_layer(review_setup):
    viewer, widget, _ = review_setup
    viewer.layers["empty labels"].visible = False

    widget.run_review()
    assert viewer.layers["empty labels"].visible
    widget.finish_review(ask=False)

    assert not viewer.layers["empty labels"].visible
    assert viewer.layers["image"].visible


def test_finish_writes_time_spent_on_last_slice(review_setup):
    viewer, widget, _ = review_setup
    widget.run_review()
    datamanager = widget.session.datamanager
    datamanager.start_time -= timedelta(minutes=5)  # 5 min on the slice, no change

    widget.finish_review(ask=False)

    assert pd.read_csv(datamanager.csv_path).at[0, "time"].startswith("00:05:")
