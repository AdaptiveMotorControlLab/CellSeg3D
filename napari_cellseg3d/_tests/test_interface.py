from napari_cellseg3d.interface import AnisotropyWidgets, Log


def test_log(qtbot):
    log = Log()
    log.print_and_log("test")

    assert log.toPlainText() == "\ntest"

    log.replace_last_line("test2")

    assert log.toPlainText() == "\ntest2"

    qtbot.add_widget(log)


def test_zoom_factor():
    resolution = [5.0, 10.0, 5.0]
    zoom = AnisotropyWidgets.anisotropy_zoom_factor(resolution)
    assert zoom == [1, 0.5, 1]
