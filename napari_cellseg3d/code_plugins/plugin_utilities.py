"""Central plugin for all utilities."""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import napari

# Qt
from qtpy.QtCore import qInstallMessageHandler
from qtpy.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

# local
import napari_cellseg3d.interface as ui
from napari_cellseg3d import utils
from napari_cellseg3d.code_plugins.plugin_base import BasePluginUtils
from napari_cellseg3d.code_plugins.plugin_convert import (
    AnisoUtils,
    ArtifactRemovalUtils,
    FragmentUtils,
    RemoveSmallUtils,
    StatsUtils,
    ThresholdUtils,
    ToInstanceUtils,
    ToSemanticUtils,
)
from napari_cellseg3d.code_plugins.plugin_crf import CRFWidget
from napari_cellseg3d.code_plugins.plugin_crop import Cropping
from napari_cellseg3d.utils import LOGGER as logger

# NOTE : to add a new utility: add it to the dictionary below, in attr_names in the Utilities class, and import it above

UTILITIES_WIDGETS = {
    "Crop": Cropping,
    "Fragment 3D volume": FragmentUtils,
    "Correct anisotropy": AnisoUtils,
    "Remove small objects": RemoveSmallUtils,
    "Convert to instance labels": ToInstanceUtils,
    "Convert to semantic labels": ToSemanticUtils,
    "Threshold": ThresholdUtils,
    "CRF": CRFWidget,
    "Label statistics": StatsUtils,
    "Clear large labels": ArtifactRemovalUtils,
}


class Utilities(QWidget, metaclass=ui.QWidgetSingleton):
    """Central plugin for all utilities."""

    def __init__(self, viewer: "napari.viewer.Viewer"):
        """Creates a widget with all utilities."""
        super().__init__()
        self._viewer = viewer
        self.current_widget = None

        attr_names = [
            "crop",
            "frag",
            "aniso",
            "small",
            "inst",
            "sem",
            "thresh",
            "crf",
            "stats",
            "artifacts",
        ]
        self._create_utils_widgets(attr_names)
        self.utils_choice = ui.DropdownMenu(
            UTILITIES_WIDGETS.keys(), text_label="Utilities"
        )

        self._build()

        self.utils_choice.currentIndexChanged.connect(self._update_visibility)
        self.utils_choice.currentIndexChanged.connect(
            self._update_current_widget
        )
        # self._dock_util()
        self._update_visibility()
        qInstallMessageHandler(ui.handle_adjust_errors_wrapper(self))

    def _update_current_widget(self):
        self.current_widget = self.utils_widgets[
            self.utils_choice.currentIndex()
        ]

    def _update_results_path(self, widget):
        self.results_filewidget.text_field.setText(str(widget.save_path))

    def _build(self):
        layout = QVBoxLayout()
        ui.add_widgets(layout, self.utils_widgets)
        ui.GroupedWidget.create_single_widget_group(
            "Utilities",
            widget=self.utils_choice,
            layout=layout,
            alignment=ui.BOTT_AL,
        )

        # layout.addWidget(self.utils_choice.label, alignment=ui.BOTT_AL)
        # layout.addWidget(self.utils_choice, alignment=ui.BOTT_AL)

        # layout.setSizeConstraint(QLayout.SetFixedSize)
        self.setLayout(layout)
        # self.setMinimumHeight(2000)
        self.setSizePolicy(
            QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding
        )
        self._update_visibility()

    def _create_utils_widgets(self, names):
        for key, name in zip(UTILITIES_WIDGETS, names):
            logger.debug(f"Creating {name} widget")
            setattr(self, name, UTILITIES_WIDGETS[key](self._viewer))

        self.utils_widgets = []
        for n in names:
            wid = getattr(self, n)
            self.utils_widgets.append(wid)

        self.current_widget = self.utils_widgets[0]
        if len(self.utils_widgets) != len(UTILITIES_WIDGETS.keys()):
            raise RuntimeError(
                "One or several utility widgets are missing/erroneous"
            )

    def _update_layers(self, current_loader, new_loader):
        current_layer = current_loader.layer()
        if not isinstance(current_layer, new_loader.layer_type):
            return
        if (
            current_layer is not None
            and current_layer.name in new_loader.layer_list.get_items()
        ):
            index = new_loader.layer_list.get_items().index(current_layer.name)
            logger.debug(
                f"Index of layer {current_layer.name} in new loader : {index}"
            )
            new_loader.layer_list.setCurrentIndex(index)

    def _update_fields(self, widget: BasePluginUtils):
        try:
            # checks all combinations to find if a layer could be recovered across widgets
            # correctness is ensured by the types of the layer loaders
            self._update_layers(
                self.current_widget.image_layer_loader,
                widget.image_layer_loader,
            )
            self._update_layers(
                self.current_widget.image_layer_loader,
                widget.label_layer_loader,
            )
            self._update_layers(
                self.current_widget.label_layer_loader,
                widget.image_layer_loader,
            )
            self._update_layers(
                self.current_widget.label_layer_loader,
                widget.label_layer_loader,
            )
        except KeyError:
            pass

        logger.debug(
            f"Current widget save path : {self.current_widget.save_path}"
        )
        logger.debug(
            f"Current widget text field : {self.current_widget.results_filewidget.text_field.text()}"
        )
        logger.debug(
            f"Matching : {self.current_widget.results_filewidget.text_field.text() == self.current_widget.results_path}"
        )
        if len(self.current_widget.utils_default_paths) > 1:
            try:
                path = self.current_widget.utils_default_paths
                default = utils.parse_default_path(path)
                widget.results_filewidget.text_field.setText(default)
                widget.utils_default_paths.append(default)
            except AttributeError:
                pass

    def _update_visibility(self):
        widget_class = UTILITIES_WIDGETS[self.utils_choice.currentText()]
        # print("vis. updated")
        # print(self.utils_widgets)
        self._hide_all()
        widget = None
        for _i, w in enumerate(self.utils_widgets):
            if isinstance(w, widget_class):
                w.setVisible(True)
                w.adjustSize()
                widget = w
            # else:
            #     self.utils_widgets[i].setVisible(False)
        self._update_fields(widget)
        self.current_widget = widget

    def _hide_all(self):
        for w in self.utils_widgets:
            w.setVisible(False)
        # self.setWindowState(Qt.WindowMaximized)
        # if self.parent() is not None:
        # print(self.parent().windowState())
        # print(int(self.parent().parent().windowState()))
        # self.parent().parent().showNormal()
        # self.parent().parent().showMaximized()
        # state = int(self.parent().parent().windowState())
        # if state == 0:
        # self.parent().parent().adjustSize()
        # elif state == 2:
        # self.parent().parent().showNormal()
        # self.parent().parent().showMaximized()
        # pass

    # def _dock_util(self):
    #     for i in range(len(self.utils_widgets)):
    #         docked = self._viewer.window.add_dock_widget(
    #             widget=self.utils_widgets[i]
    #         )
    #         self.docked_widgets.append(docked)

    # def remove_from_viewer(self):
    #     self._viewer.window.remove_dock_widget(self)
