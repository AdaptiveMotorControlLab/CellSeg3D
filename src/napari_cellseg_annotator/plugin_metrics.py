from napari_cellseg_annotator.plugin_base import BasePluginFolder
from napari_cellseg_annotator import interface as ui
from napari_cellseg_annotator import utils
from tifffile import imread

from monai.transforms import ToTensor, AsDiscrete, EnsureChannelFirst, SpatialPad, Compose, AddChannel, Orientation
from monai.metrics import DiceMetric


class MetricsUtils(BasePluginFolder):

    def __init__(self, viewer: "napari.viewer.Viewer", parent):

        super().__init__(viewer, parent)

        self._viewer = viewer

        self.btn_compute_dice = ui.make_button("Compute Dice",self.compute_dice)

        self.btn_result_path.setVisible(False)
        self.lbl_result_path.setVisible(False)

        self.build()

    def build(self):


        self.lbl_filetype.setVisible(False)

        w, layout = ui.make_container_widget()

        metrics_group_w,metrics_group_l = ui.make_group("Metrics")

        self.lbl_image_files.setText("Ground truth")

        metrics_group_l.addWidget(ui.combine_blocks(
                second=self.btn_image_files,
                first=self.lbl_image_files,
                min_spacing=70,
            ),
            alignment=ui.LEFT_AL,)

        self.lbl_label_files.setText("Prediction")

        metrics_group_l.addWidget(ui.combine_blocks(
            second=self.btn_label_files,
            first=self.lbl_label_files,
            min_spacing=70,
        ),
            alignment=ui.LEFT_AL, )

        metrics_group_l.addWidget(self.btn_compute_dice, alignment = ui.LEFT_AL)

        metrics_group_w.setLayout(metrics_group_l)
        layout.addWidget(metrics_group_w)

        ui.make_scrollable(layout, self)


    def compute_dice(self):
        transforms = Compose(ToTensor(), Orientation(axcodes = "PLI",image_only=True), EnsureChannelFirst(), AsDiscrete(threshold=0.5))

        for ground_path, pred_path in zip(self.images_filepaths, self.labels_filepaths):

            ground = imread(ground_path)
            pred = imread(pred_path)

            pad = utils.get_padding_dim(pred.shape[-3:-1])

            ground = transforms(ground)
            pred = transforms(pred)

            ground = AddChannel()(ground)
            ground = SpatialPad(pad)(ground)
            ground = AddChannel()(ground)

            print(ground.shape)
            print(pred.shape)

            score = DiceMetric()(ground, pred)
            print(score)