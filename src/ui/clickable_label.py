from pathlib import Path

from PIL import Image as Image
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QCursor, QPixmap, QTransform
from PyQt5.QtWidgets import QFrame, QHBoxLayout, QLabel


def clip(val: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    return max(min(val, max_val), min_val)


class ImagePairLayout(QHBoxLayout):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.im_left = ClickableImageLabel(align_left=False)
        self.im_right = ClickableImageLabel(align_left=True)
        self.addWidget(self.im_left)
        self.addWidget(self.im_right)

    def _set_optimal_dimensions_from_pixmaps(
        self, qp1: QPixmap, qp2: QPixmap, *, min_width_frac: float = 1 / 3
    ) -> tuple[QSize, QSize]:
        parent_width = self.parent().width()
        parent_max_height = self.parent().maximumSize().height()

        scaled_w1 = qp1.width() * parent_max_height / qp1.height()
        scaled_w2 = qp2.width() * parent_max_height / qp2.height()
        # apply a "scaling penalty" that interpolated between the scaled width and the unscaled width
        scaled_w1 = 0.5 * (scaled_w1 + qp1.width())
        scaled_w2 = 0.5 * (scaled_w2 + qp2.width())

        width1 = round(
            clip(
                scaled_w1 / (scaled_w1 + scaled_w2),
                min_width_frac,
                1 - min_width_frac,
            )
            * parent_width
        )
        width2 = parent_width - width1
        return QSize(width1, parent_max_height), QSize(width2, parent_max_height)

    def set_images(self, img_left: Path, rot1: int, img_right: Path, rot2: int):
        pixmap1 = QPixmap(str(img_left.absolute())).transformed(
            QTransform().rotate(rot1)
        )
        pixmap2 = QPixmap(str(img_right.absolute())).transformed(
            QTransform().rotate(rot2)
        )
        print("raw pixmaps", pixmap1.size(), pixmap2.size())
        size1, size2 = self._set_optimal_dimensions_from_pixmaps(pixmap1, pixmap2)
        print("fitted", size1, size2)

        self.im_left.setFixedSize(size1)
        # self.im_left.setSizePolicy(
        #     QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # )
        self.im_left.change_image_to(img_left)
        self.im_right.setFixedSize(size2)
        # self.im_right.setSizePolicy(
        #     QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # )
        self.im_right.change_image_to(img_right)

    def get_images(self) -> tuple[Path, Path]:
        return self.im_left.original_pixmap, self.im_right.original_pixmap


class ClickableImageLabel(QLabel):
    def __init__(self, parent=None, *, align_left: bool = False):
        super().__init__(parent)
        self.setCursor(QCursor(Qt.PointingHandCursor))
        self.setFrameStyle(QLabel.StyledPanel | QFrame.Plain)
        self.setLineWidth(3)
        self.original_pixmap = None
        self.is_hovered = False
        self.rotation = 0
        self.alignment = Qt.AlignLeft if align_left else Qt.AlignRight

    @staticmethod
    def get_base_rotation(img: Path) -> int:
        with Image.open(img) as img:
            try:
                exif = img._getexif()
            except AttributeError:
                return 0
            if not exif:
                return 0
            match exif.get(274):
                case 3:
                    return 180
                case 6:
                    return 90
                case 8:
                    return 270
        return 0

    def change_image_to(self, path: Path):
        base_pixmap = QPixmap(str(path.absolute()))
        pixmap = base_pixmap.transformed(
            QTransform().rotate(self.get_base_rotation(path))
        )
        self.original_pixmap = pixmap
        self.setPixmap(
            pixmap.scaled(
                self.width(),
                self.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
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
