from tifffile import imread
from pathlib import Path
import warnings

from qtpy.QtWidgets import QTextEdit

from napari_cellseg3d.code_plugins.plugin_model_inference import Inferer


class LogFixture(QTextEdit):
    def __init__(self):
        super(LogFixture, self).__init__()

    def print_and_log(self, text, printing=None):
        print(text)

    def warn(self, warning):
        warnings.warn(warning)


def test_inference(make_napari_viewer, qtbot):

    im_path = str(Path(__file__).resolve().parent / "res/test.tif")
    image = imread(im_path)

    assert image.shape == (6, 6, 6)

    viewer = make_napari_viewer()
    widget = Inferer(viewer)
    widget.log = LogFixture()
    viewer.window.add_dock_widget(widget)
    viewer.add_image(image)

    assert len(viewer.layers) == 1

    widget.window_infer_box.setChecked(True)
    widget.window_overlap_slider.setValue(0.0)
    widget.keep_data_on_cpu_box.setChecked(True)

    assert widget.check_ready()

    widget.start()
    assert widget.worker is not None

    with qtbot.waitSignal(signal=widget.worker.finished) as blocker:
        blocker.connect(widget.worker.errored)

    assert len(viewer.layers) == 2
