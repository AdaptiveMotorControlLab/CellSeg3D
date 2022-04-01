import napari
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    EnsureTyped,
    EnsureType,
    LabelFilter,
)
from napari_cellseg_annotator import utils
from napari_cellseg_annotator.model_framework import ModelFramework
from qtpy.QtWidgets import (
    QVBoxLayout,
    QPushButton,
    QSizePolicy,
    QLabel,
    QCheckBox,
    QComboBox,
)


class Inferer(ModelFramework):
    """A plugin to run already trained models in evaluation mode to preform inference and output a volume label."""

    def __init__(self, viewer: "napari.viewer.Viewer"):

        super().__init__(viewer)

        self.models_dict = {"VNet": " ", "SegResNet": " "}
        self.current_model = None

        self.view_checkbox = QCheckBox()
        self.view_checkbox.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.lbl_view = QLabel("View in napari after prediction ?", self)

        self.model_choice = QComboBox()
        self.model_choice.addItems(sorted(self.models_dict.keys()))
        self.lbl_model_choice = QLabel("Model name", self)

        self.btn_start = QPushButton("Start inference")
        self.btn_start.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn_start.clicked.connect(self.start)

        self.btn_label_files.setVisible(False)
        self.lbl_label_files.setVisible(False)
        self.btn_model_path.setVisible(False)
        self.lbl_model_path.setVisible(False)

        self.build()

    def get_model(self, key):
        return self.models_dict[key]

    def build(self):

        vbox = QVBoxLayout()

        vbox.addWidget(
            utils.combine_blocks(self.filetype_choice, self.lbl_filetype)
        )  # file extension
        vbox.addWidget(
            utils.combine_blocks(self.btn_image_files, self.lbl_image_files)
        )  # in folder
        vbox.addWidget(
            utils.combine_blocks(self.btn_result_path, self.lbl_result_path)
        )  # out folder

        vbox.addWidget(
            utils.combine_blocks(self.model_choice, self.lbl_model_choice)
        )  # model choice
        vbox.addWidget(
            utils.combine_blocks(self.view_checkbox, self.lbl_view)
        )  # view_after bool

        # TODO : add custom model handling ? using exec() to read user provided model class
        # self.lbl_label.setText("model.pth directory :")

        vbox.addWidget(self.btn_start)
        vbox.addWidget(self.btn_close)

        self.setLayout(vbox)

    def start(self):

        post_process_transforms = Compose(
            EnsureType(),
            AsDiscrete(threshold=0.1),
            LabelFilter(applied_labels=[0]),
        )

        load_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                # AddChanneld(keys=["image"]), #already done
                EnsureChannelFirstd(keys=["image"]),
                EnsureTyped(keys=["image"]),
            ]
        )
        return
