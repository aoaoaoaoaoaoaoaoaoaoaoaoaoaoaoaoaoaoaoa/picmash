from pathlib import Path

from PIL import Image as Image
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QCursor, QPixmap, QTransform
from PyQt5.QtWidgets import QLabel


class ClickableImageLabel(QLabel):
    def __init__(self, parent=None, *, align_left: bool = False):
        super().__init__(parent)
        self.setCursor(QCursor(Qt.PointingHandCursor))
        self.original_pixmap = None
        self.is_hovered = False
        self.rotation = 0
        self.alignment = Qt.AlignLeft if align_left else Qt.AlignRight

    @staticmethod
    def get_base_rotation(img: Path):
        with Image.open(img) as img:
            exif = img._getexif()
            if exif:
                orientation = exif.get(274)
                if orientation == 3:
                    return 180
                elif orientation == 6:
                    return 90
                elif orientation == 8:
                    return 270
        return 0

    def change_image_to(self, path: Path):
        pixmap = QPixmap(str(path.absolute())).transformed(
            QTransform().rotate(self.get_base_rotation(path))
        )
        self.original_pixmap = pixmap
        self.setPixmap(
            pixmap.scaled(
                self.width(), self.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            ),
        )
        self.rotation = 0
        self.setAlignment(self.alignment)

    def mousePressEvent(self, event):
        self.parent().image_clicked(self)

    def enterEvent(self, event):
        self.is_hovered = True

    def leaveEvent(self, event):
        self.is_hovered = False

    def rotate_image(self, rotation: int = 90):
        if self.original_pixmap and self.is_hovered:
            self.rotation = (self.rotation + rotation) % 360
            transform = QTransform().rotate(self.rotation)
            rotated_pixmap = self.original_pixmap.transformed(
                transform, Qt.SmoothTransformation
            )
            self.setPixmap(
                rotated_pixmap.scaled(
                    self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                ),
            )
            self.setAlignment(self.alignment)
