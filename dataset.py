"""Load paired RGB images and binary masks from the fixed Kvasir-SEG split."""

import csv
from pathlib import Path, PurePosixPath
import random

import cv2
import numpy as np


IMAGE_SIZE = (256, 256)
DEFAULT_SPLITS = Path(__file__).resolve().with_name("splits.csv")


class KvasirDataset:
    """Return an RGB uint8 image (H, W, 3) and a 0/1 uint8 mask (H, W).

    Select a split explicitly. Use augment=True for training augmentation.
    Dataset paths in the CSV are relative to the extracted Kvasir-SEG folder.
    """

    def __init__(self, dataset_dir, split, *, csv_path=DEFAULT_SPLITS, augment=False):
        if split not in {"train", "val", "test"}:
            raise ValueError("split은 'train', 'val', 'test' 중 하나여야 합니다.")
        if augment and split != "train":
            raise ValueError("증강은 학습 세트에만 적용할 수 있습니다.")

        dataset_dir = Path(dataset_dir).expanduser()
        self.augment = augment
        self.samples = []
        with Path(csv_path).expanduser().open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not {"split", "image_path", "mask_path"}.issubset(reader.fieldnames or ()):
                raise ValueError("CSV에 split, image_path, mask_path 열이 필요합니다.")

            for row in reader:
                if row["split"] != split:
                    continue
                image = PurePosixPath(row["image_path"] or "")
                mask = PurePosixPath(row["mask_path"] or "")
                if (
                    image.parts != ("images", image.name)
                    or mask.parts != ("masks", mask.name)
                    or image.name != mask.name
                    or image.suffix.lower() not in {".jpg", ".jpeg"}
                ):
                    raise ValueError(f"CSV {reader.line_num}행의 이미지·마스크 경로를 확인하세요.")
                self.samples.append((dataset_dir / image, dataset_dir / mask))

        if not self.samples:
            raise ValueError(f"CSV에서 {split} 세트에 해당하는 파일을 찾지 못했습니다.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path = self.samples[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"이미지를 읽을 수 없습니다. {image_path}")
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise OSError(f"마스크를 읽을 수 없습니다. {mask_path}")
        if image.shape[:2] != mask.shape:
            raise ValueError(f"이미지와 마스크의 원본 크기가 다릅니다. {image_path.name}")

        image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # JPEG 마스크의 밝기를 배경(0)과 용종(1)으로 변환한다.
        mask = (mask >= 128).astype(np.uint8)
        mask = cv2.resize(mask, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)

        if self.augment:
            image, mask = _augment_pair(image, mask)
        return image, mask


def _augment_pair(image, mask):
    if random.random() < 0.5:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    height, width = mask.shape
    center = ((width - 1) / 2, (height - 1) / 2)
    rotation = cv2.getRotationMatrix2D(center, random.uniform(-10, 10), 1)
    image = cv2.warpAffine(image, rotation, (width, height), flags=cv2.INTER_LINEAR)
    mask = cv2.warpAffine(mask, rotation, (width, height), flags=cv2.INTER_NEAREST)
    return image, mask
