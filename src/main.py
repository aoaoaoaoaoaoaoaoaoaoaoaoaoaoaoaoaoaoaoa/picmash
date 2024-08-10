import logging
import random
import sqlite3
import sys
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from sqlite3 import connect
from typing import Self, final

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)
from peewee import CharField, FloatField, IntegerField, Model, SqliteDatabase
from xdg_base_dirs import xdg_config_home

from src.ui.clickable_label import ClickableImageLabel

# TODO make configurable
# temperature for image selection
IMG_SEL_TEMP = 0.5
IMG_SEL_UNSEEN_PREF = 0.5
ELO_K = 0.15

RANKS_DB = xdg_config_home() / "image_ranks.db"
DB = SqliteDatabase(RANKS_DB)


@final
class ImgWrapper(Model):
    class Meta:
        database = DB
        table_name = "image_scores"

    checksum = CharField(primary_key=True)
    path = CharField()
    score = FloatField(default=0.0)
    rotation = IntegerField(default=0)

    @classmethod
    def from_path(cls, path: Path) -> Self:
        checksum = sha256(path.read_bytes()).hexdigest()
        return cls.get_or_create(checksum=checksum, path=str(path.absolute()))[0]

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
    fst: ImgWrapper
    snd: ImgWrapper

    def update_ratings_inplace(self, *, fst_won: bool):
        """
        Updates the ratings of the images in the pair in place.
        """
        r1 = 10**self.fst.score
        r2 = 10**self.snd.score
        e1 = r1 / (r1 + r2)
        e2 = r2 / (r1 + r2)

        if fst_won:
            self.fst.score += ELO_K * (1 - e1)
            self.snd.score += ELO_K * (0 - e2)
        else:
            self.fst.score += ELO_K * (0 - e1)
            self.snd.score += ELO_K * (1 - e2)


# noinspection PyPropertyAccess
class ImageRanker(QWidget):
    def __init__(self):
        super().__init__()
        self.init_db()
        self.images: list[ImgWrapper] = []
        self.init_ui()
        self.load_images()
        self.cur_pair = self.sample_pair(self.images)

        self.draw_cur_pair()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_R:
            if self.img1_label.is_hovered:
                self.img1_label.rotate_image()
                self.cur_pair.fst.rotation = (self.cur_pair.fst.rotation + 90) % 360
            elif self.img2_label.is_hovered:
                self.img2_label.rotate_image()
                self.cur_pair.snd.rotation = (self.cur_pair.snd.rotation + 90) % 360
        # ctrl+d deletes hovered image
        elif event.key() == Qt.Key_Delete:
            self.delete_hovered_image()

    @property
    def hovered_image(self) -> ImgWrapper | None:
        if self.img1_label.is_hovered:
            return self.cur_pair.fst
        elif self.img2_label.is_hovered:
            return self.cur_pair.snd
        else:
            return None

    def delete_hovered_image(self):
        img_to_delete = self.hovered_image
        if not img_to_delete:
            return
        logging.info(f"Deleting image {img_to_delete}")
        with connect(RANKS_DB) as conn:
            conn.execute(
                "DELETE FROM image_scores WHERE checksum = ?", (img_to_delete.checksum,)
            )
        self.images.remove(img_to_delete)
        img_to_delete.path.unlink()
        if self.hovered_image is self.cur_pair.fst:
            self.cur_pair = DuelingPair(self.sample_one(self.images), self.cur_pair.snd)
        elif self.hovered_image is self.cur_pair.snd:
            self.cur_pair = DuelingPair(self.cur_pair.fst, self.sample_one(self.images))
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
        self.setGeometry(100, 100, 1920, 1080)

        main_layout = QVBoxLayout()
        image_layout = QHBoxLayout()

        self.img1_label = ClickableImageLabel(self, align_left=False)
        self.img2_label = ClickableImageLabel(self, align_left=True)
        image_layout.addWidget(self.img1_label)
        image_layout.addWidget(self.img2_label)

        main_layout.addLayout(image_layout)
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
        for img, label in zip(
            [self.cur_pair.fst, self.cur_pair.snd], [self.img1_label, self.img2_label]
        ):
            label.change_image_to(img.path)
            label.rotate_image(img.rotation)

    def sample_and_draw_pair(self):
        self.cur_pair = self.sample_pair(self.images)
        self.draw_cur_pair()

    def image_clicked(self, label):
        self.cur_pair.update_ratings_inplace(fst_won=label == self.img1_label)
        self.sample_and_draw_pair()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.draw_cur_pair()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = QApplication(sys.argv)
    ex = ImageRanker()
    ex.show()
    sys.exit(app.exec_())
