from napari_cellseg3d.code_plugins.plugin_utilities import Utilities
from napari_cellseg3d.code_plugins.plugin_utilities import UTILITIES_WIDGETS


def test_utils_plugin(make_napari_viewer):
    view = make_napari_viewer()
    widget = Utilities(view)

    view.window.add_dock_widget(widget)
    for i, utils_name in enumerate(UTILITIES_WIDGETS.keys()):
        widget.utils_choice.setCurrentIndex(i)
        assert isinstance(
            widget.utils_widgets[i], UTILITIES_WIDGETS[utils_name]
        )
