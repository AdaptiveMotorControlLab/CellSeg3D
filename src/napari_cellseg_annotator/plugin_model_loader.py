import napari
from napari_cellseg_annotator import utils
from qtpy.QtWidgets import (
    QTabWidget,
    QVBoxLayout,
    QPushButton,
    QSizePolicy,
    QLabel,
    QComboBox,
)


class Model_Loader(QTabWidget):
    def __init__(self, viewer: "napari.viewer.Viewer", parent=None):

        super().__init__(parent)

        # self.master = parent
        self._viewer = viewer
        """napari.viewer.Viewer: viewer in which the widget is displayed"""

        self.data_path = ""
        self.label_path = ""
        self.results_path = ""
        self._default_path = [self.data_path, self.label_path]

        # data dir
        self.btn_dat = QPushButton("Open", self)
        self.btn_dat.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn_dat.clicked.connect(self.show_dialog_dat)

        self.lbl_dat = QLabel("Dataset directory :")

        # label dir
        self.btn_label = QPushButton("Open", self)
        self.btn_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn_label.clicked.connect(self.show_dialog_lab)

        self.lbl_label = QLabel("Labels directory :", self)

        # filetype choice
        self.filetype_choice = QComboBox()
        self.filetype_choice.addItems([".tif", ".png"])
        self.filetype_choice.setSizePolicy(
            QSizePolicy.Fixed, QSizePolicy.Fixed
        )
        self.lblft_expl = QLabel(
            "(Folders of .png or single .tif files)", self
        )
        self.lblft = QLabel("Filetype :", self)

        # model choice
        self.model_choice = QComboBox()
        # TODO : add models
        self.model_choice.addItems(["MODELS"])
        self.model_choice.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.lbl_mod = QLabel("Filetype :", self)

    def build(self):

        vbox = QVBoxLayout()

        vbox.addWidget(utils.combine_blocks(self.btn_dat, self.lbl_dat))
        vbox.addWidget(utils.combine_blocks(self.btn_label, self.lbl_label))
        vbox.addWidget(utils.combine_blocks(self.filetype_choice, self.lblft))
        vbox.addWidget(self.lblft_expl)
        vbox.addWidget(self.model_choice)

    def show_dialog_lab(self):
        f_name = utils.open_file_dialog(self, self._default_path)

        if f_name:
            self.label_path = f_name
            self.lbl_label.setText(self.label_path)

    def show_dialog_dat(self):
        f_name = utils.open_file_dialog(self, self._default_path)

        if f_name:
            self.data_path = f_name
            self.lbl_dat.setText(self.label_path)

    def select_model(self):
        return

    def close(self):
        """Close the widget"""
        self._viewer.window.remove_dock_widget(self)
