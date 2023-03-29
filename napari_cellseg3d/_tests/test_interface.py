from napari_cellseg3d.interface import Log


def test_log(qtbot):
    log = Log()
    log.print_and_log("test")

    assert log.toPlainText() == "\ntest"

    log.replace_last_line("test2")

    assert log.toPlainText() == "\ntest2"

    qtbot.add_widget(log)
