import napari
from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton
from napari_cellseg_annotator import utils


class Helper(QWidget):
    # widget testing
    def __init__(self, parent: "napari.viewer.Viewer"):
        super().__init__()
        # self.master = parent
        self.help_url = (
            "https://github.com/C-Achard/cellseg-annotator-test/tree/main"
        )

        self.about_url = "https://wysscenter.ch/advances/3d-computer-vision-for-brain-analysis"
        self._viewer = parent
        self.btn1 = QPushButton("Help...", self)
        # self.btn1.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn1.clicked.connect(lambda: utils.open_url(self.help_url))
        self.btn2 = QPushButton("About...", self)
        # self.btn2.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn2.clicked.connect(lambda: utils.open_url(self.about_url))
        self.btnc = QPushButton("Close", self)
        # self.btnc.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btnc.clicked.connect(self.close)
        self.build()

    def build(self):
        vbox = QVBoxLayout()
        vbox.addWidget(self.btn1)
        vbox.addWidget(self.btn2)
        vbox.addWidget(self.btnc)
        self.setLayout(vbox)
        self.show()

    def close(self):
        self._viewer.window.remove_dock_widget(self)
