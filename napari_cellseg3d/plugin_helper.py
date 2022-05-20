import napari

# Qt
from qtpy.QtWidgets import QSizePolicy
from qtpy.QtWidgets import QVBoxLayout
from qtpy.QtWidgets import QWidget

# local
from napari_cellseg3d import interface as ui


class Helper(QWidget):
    # widget testing
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()

        self.help_url = "https://adaptivemotorcontrollab.github.io/cellseg3d-docs/"

        self.about_url = "https://wysscenter.ch/advances/3d-computer-vision-for-brain-analysis"
        self._viewer = viewer

        self.info_label = ui.make_label(f"napari-cellseg3d v.{0.01}", self)

        self.btn1 = ui.make_button(
            "Help...", lambda: ui.open_url(self.help_url)
        )
        self.btn2 = ui.make_button(
            "About...", lambda: ui.open_url(self.about_url)
        )
        self.btnc = ui.make_button("Close", self.remove_from_viewer)

        self.build()

    def build(self):
        vbox = QVBoxLayout()

        widgets = [
            self.btn1,
            self.btn2,
            self.btnc,
        ]
        #################
        if self.test:
            widgets.append(self.epoch)
        #################
        ui.add_widgets(vbox, widgets)
        self.setLayout(vbox)
        # self.show()
        # self._viewer.window.add_dock_widget(self, name="Help/About...", area="right")

    def remove_from_viewer(self):
        self._viewer.window.remove_dock_widget(self)
