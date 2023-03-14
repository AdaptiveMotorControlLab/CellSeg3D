from napari_cellseg3d.code_plugins.plugin_helper import Helper


def test_helper(make_napari_viewer):

    viewer = make_napari_viewer()
    widget = Helper(viewer)

    dock = viewer.window.add_dock_widget(widget)
    children = len(viewer.window._dock_widgets)

    assert dock is not None

    widget.btnc.click()

    assert len(viewer.window._dock_widgets) == children - 1
