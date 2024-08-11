import logging
import random
import sqlite3
import sys
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Self, final

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from peewee import CharField, FloatField, IntegerField, Model, SqliteDatabase
from xdg_base_dirs import xdg_config_home

from src.ui.clickable_label import ClickableImageLabel, ImagePairLayout

# TODO make configurable
# temperature for image selection
IMG_SEL_TEMP = 0.5
IMG_SEL_UNSEEN_PREF = 0.5
ELO_K = 0.15

RANKS_DB = xdg_config_home() / "image_ranks.db"
DB = SqliteDatabase(RANKS_DB)


class PathField(CharField):
    def db_value(self, value: Path) -> str:
        return str(value.absolute())

    def python_value(self, value: str) -> Path:
        return Path(value).absolute()


@final
class ImgWrapper(Model):
    class Meta:
        database = DB
        table_name = "image_scores"

    checksum = CharField(primary_key=True)
    path = PathField()
    score = FloatField(default=0.0)
    rotation = IntegerField(default=0)

    @classmethod
    def from_path(cls, path: Path) -> Self:
        checksum = sha256(path.read_bytes()).hexdigest()
        return cls.get_or_create(checksum=checksum, path=path.absolute())[0]

    @property
    def sample_weight(self) -> float:
        return 10 ** (
            ((IMG_SEL_UNSEEN_PREF if self.score == 0.0 else 0.0) + self.score)
            / IMG_SEL_TEMP
        )

    def __hash__(self):
        return hash(self.checksum)

    def __str__(self):
        return f"Img[{self.checksum[-8:]}({1000 * self.score:.2f})]"


# noinspection PyPropertyAccess
@final
@dataclass
class DuelingPair:
    left: ImgWrapper
    right: ImgWrapper

    def update_ratings_inplace(self, *, left_won: bool):
        """
        Updates the ratings of the images in the pair in place.
        """
        r1 = 10**self.left.score
        r2 = 10**self.right.score
        e1 = r1 / (r1 + r2)
        e2 = r2 / (r1 + r2)

        if left_won:
            self.left.score += ELO_K * (1 - e1)
            self.right.score += ELO_K * (0 - e2)
        else:
            self.left.score += ELO_K * (0 - e1)
            self.right.score += ELO_K * (1 - e2)

        self.left.save()
        self.right.save()


@dataclass
class ImgWithLabel:
    label: ClickableImageLabel
    img: ImgWrapper


# noinspection PyPropertyAccess
class ImageRanker(QWidget):
    @staticmethod
    def get_main_stylesheet() -> str:
        return """
        QMainWindow {
            background-color: #111;
        }
        ClickableImageLabel {
            background-color: #111;
        }
        """

    @property
    def left(self) -> ImgWithLabel:
        return ImgWithLabel(self.pair_layout.im_left, self.cur_pair.left)

    @property
    def right(self) -> ImgWithLabel:
        return ImgWithLabel(self.pair_layout.im_right, self.cur_pair.right)

    @property
    def hovered(self) -> ImgWithLabel | None:
        if self.left.label.is_hovered:
            return self.left
        elif self.right.label.is_hovered:
            return self.right
        else:
            return None

    def __init__(self):
        super().__init__()
        self.setStyleSheet(self.get_main_stylesheet())
        self.init_db()
        self.images: list[ImgWrapper] = []
        self.pair_layout = ImagePairLayout(self)
        self.init_ui()
        self.load_images()
        self.cur_pair = self.sample_pair(self.images)

        self.draw_cur_pair()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_R:
            self.hovered.label.rotate_image()
            self.hovered.img.rotation = self.hovered.label.rotation
            self.hovered.img.save()
        # ctrl+d deletes hovered image
        elif event.key() == Qt.Key_Delete:
            self.delete_hovered_image()

    def delete_hovered_image(self):
        hovered = self.hovered
        if not hovered:
            return
        hovered.img.delete_instance()
        self.images.remove(hovered.img)
        hovered.img.path.unlink()
        if self.hovered.img is self.cur_pair.left:
            self.cur_pair = DuelingPair(
                self.sample_one(self.images), self.cur_pair.right
            )
        elif self.hovered.img is self.cur_pair.right:
            self.cur_pair = DuelingPair(
                self.cur_pair.left, self.sample_one(self.images)
            )
        self.draw_cur_pair()

    @staticmethod
    def init_db():
        with sqlite3.connect(RANKS_DB) as conn:
            conn.execute(
                """
                    CREATE TABLE IF NOT EXISTS image_scores
                    (checksum TEXT PRIMARY KEY, path TEXT, score REAL)
                """
            )

    def init_ui(self):
        self.setWindowTitle("Image Ranker")
        w, h = 1920, 1080
        self.setGeometry(100, 100, w, h)
        self.setMaximumSize(w, h)
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)

        main_layout = QVBoxLayout()
        main_layout.addLayout(self.pair_layout)
        self.setLayout(main_layout)

        # TODO store paths in config
        self.root_dir = Path(
            QFileDialog.getExistingDirectory(self, "Select Image Folder")
        )

        if not self.root_dir:
            print("No folder selected. Exiting.")
            sys.exit()

    def populate_scores_with_ratings(self):
        for wrapper in ImgWrapper.select():
            self.images.append(wrapper)

    def load_images(self):
        for path in self.root_dir.glob("**/*"):
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".gif"}:
                continue
            img = ImgWrapper.from_path(path)
            self.images.append(img)

    @classmethod
    def sample_one(cls, images: list[ImgWrapper]) -> ImgWrapper:
        return random.choices(images, weights=[img.sample_weight for img in images])[0]

    @classmethod
    def sample_pair(cls, images: list[ImgWrapper]) -> DuelingPair:
        while (fst := cls.sample_one(images)) == (snd := cls.sample_one(images)):
            pass
        logging.info(f"Sampled pair: {fst.path} vs {snd.path}")
        return DuelingPair(fst, snd)

    def draw_cur_pair(self) -> None:
        self.pair_layout.set_images(
            self.cur_pair.left.path,
            self.cur_pair.left.rotation,
            self.cur_pair.right.path,
            self.cur_pair.right.rotation,
        )

    def sample_and_draw_pair(self):
        self.cur_pair = self.sample_pair(self.images)
        self.draw_cur_pair()

    def image_clicked(self, label: ClickableImageLabel):
        self.cur_pair.update_ratings_inplace(left_won=label == self.left.label)
        self.sample_and_draw_pair()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = QApplication(sys.argv)
    ex = ImageRanker()
    ex.show()
    sys.exit(app.exec_())
