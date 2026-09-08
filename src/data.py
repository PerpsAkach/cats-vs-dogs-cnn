from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

IMAGE_SIZE = (224, 224)
CLASS_MAP = {"cats": 0, "dogs": 1}
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def discover_images(root: Path):
    paths, labels = [], []
    for class_name, label in CLASS_MAP.items():
        class_dir = root / class_name
        if not class_dir.exists():
            raise FileNotFoundError(class_dir)
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
                paths.append(path)
                labels.append(label)
    return paths, labels


def load_image(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize(IMAGE_SIZE)
    return np.asarray(image, dtype=np.float32) / 255.0


def build_arrays(root: Path):
    paths, labels = discover_images(root)
    x = np.stack([load_image(p) for p in paths])
    y = np.asarray(labels, dtype=np.int32)
    return x, y, paths


def split_data(x, y, paths):
    return train_test_split(
        x, y, paths,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
