"""Tiny plugin showing link to documentation and about page."""
import pathlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import napari

# Qt
from qtpy.QtCore import QSize
from qtpy.QtGui import QIcon, QPixmap
from qtpy.QtWidgets import QVBoxLayout, QWidget

# local
from napari_cellseg3d import interface as ui


class Helper(QWidget, metaclass=ui.QWidgetSingleton):
    """Tiny plugin showing link to documentation and about page."""

    def __init__(self, viewer: "napari.viewer.Viewer"):
        """Creates a widget with links to documentation and about page."""
        super().__init__()

        self.help_url = "https://adaptivemotorcontrollab.github.io/CellSeg3d/"

        self.about_url = "https://wysscenter.ch/advances/3d-computer-vision-for-brain-analysis"
        self.repo_url = "https://github.com/AdaptiveMotorControlLab/CellSeg3d"
        self._viewer = viewer

        logo_path = str(
            pathlib.Path(__file__).parent.resolve() / "../res/logo_alpha.png"
        )
        print(logo_path)
        image = QPixmap(logo_path)

        self.logo_label = ui.Button(func=lambda: ui.open_url(self.repo_url))
        self.logo_label.setIcon(QIcon(image))
        self.logo_label.setMinimumSize(200, 200)
        self.logo_label.setIconSize(QSize(200, 200))
        self.logo_label.setStyleSheet(
            "QPushButton { background-color: transparent }"
        )
        self.logo_label.setToolTip("Open Github page")

        self.info_label = ui.make_label(
            f"You are using napari-cellseg3d v.{'0.1.1'}\n\n"
            f"Plugin for cell segmentation developed\n"
            f"by the Mathis Lab of Adaptive Motor Control\n\n"
            f"Code by :\nCyril Achard\nMaxime Vidal\nJessy Lauer\nMackenzie Mathis\n"
            f"\nReleased under the MIT license",
            self,
        )

        self.btn1 = ui.Button("Help...", lambda: ui.open_url(self.help_url))
        self.btn1.setToolTip("Go to documentation")

        self.btn2 = ui.Button("About...", lambda: ui.open_url(self.about_url))

        self.btnc = ui.Button("Close", self.remove_from_viewer)

        self.build()

    def build(self):
        """Build the widget.""."""
        vbox = QVBoxLayout()

        widgets = [
            self.logo_label,
            self.info_label,
            self.btn1,
            self.btn2,
            self.btnc,
        ]
        ui.add_widgets(vbox, widgets)
        self.setLayout(vbox)

    def remove_from_viewer(self):
        """Remove the widget from the viewer."""
        self._viewer.window.remove_dock_widget(self)
