from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .config import DEFAULT_CONFIG, TrainingConfig

CLASS_MAP = {"cats": 0, "dogs": 1}
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class DatasetSummary:
    root: Path
    total_images: int
    cat_images: int
    dog_images: int

    @property
    def class_balance_ratio(self) -> float:
        minority = min(self.cat_images, self.dog_images)
        majority = max(self.cat_images, self.dog_images)
        return float(minority / majority) if majority else 0.0


def discover_images(root: Path) -> tuple[list[Path], list[int]]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {root}")

    paths: list[Path] = []
    labels: list[int] = []
    for class_name, label in CLASS_MAP.items():
        class_dir = root / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        if not class_dir.is_dir():
            raise NotADirectoryError(f"Class path is not a directory: {class_dir}")

        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
                paths.append(path)
                labels.append(label)

    if not paths:
        raise ValueError("No supported images were found")
    return paths, labels


def summarize_dataset(root: Path) -> DatasetSummary:
    paths, labels = discover_images(root)
    cat_images = labels.count(CLASS_MAP["cats"])
    dog_images = labels.count(CLASS_MAP["dogs"])
    if cat_images == 0 or dog_images == 0:
        raise ValueError("Both cats and dogs classes must contain at least one image")
    return DatasetSummary(
        root=Path(root),
        total_images=len(paths),
        cat_images=cat_images,
        dog_images=dog_images,
    )


def load_image(path: Path, image_size: tuple[int, int] | None = None) -> np.ndarray:
    """Load one image as normalized RGB; image_size is (height, width)."""

    target_height, target_width = image_size or DEFAULT_CONFIG.image_size
    with Image.open(path) as image:
        rgb = image.convert("RGB").resize((target_width, target_height))
        return np.asarray(rgb, dtype=np.float32) / 255.0


def validate_images(paths: Iterable[Path]) -> list[Path]:
    """Return unreadable image paths without loading the full dataset into memory."""

    unreadable: list[Path] = []
    for path in paths:
        try:
            with Image.open(path) as image:
                image.verify()
        except (OSError, ValueError, SyntaxError):
            unreadable.append(Path(path))
    return unreadable


def build_tf_datasets(
    root: Path,
    config: TrainingConfig | None = None,
):
    """Build memory-efficient TensorFlow datasets from class directories.

    TensorFlow is imported lazily so discovery/validation utilities remain usable
    in lightweight environments.
    """

    cfg = config or DEFAULT_CONFIG
    cfg.validate()
    summary = summarize_dataset(root)

    import tensorflow as tf

    common = {
        "directory": str(Path(root)),
        "labels": "inferred",
        "label_mode": "binary",
        "class_names": ["cats", "dogs"],
        "image_size": cfg.image_size,
        "batch_size": cfg.batch_size,
        "validation_split": cfg.validation_fraction,
        "seed": cfg.random_seed,
    }
    train_ds = tf.keras.utils.image_dataset_from_directory(
        subset="training",
        shuffle=True,
        **common,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        subset="validation",
        shuffle=False,
        **common,
    )

    normalizer = tf.keras.layers.Rescaling(1.0 / 255.0)
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.map(
        lambda images, labels: (normalizer(images), labels),
        num_parallel_calls=autotune,
    ).prefetch(autotune)
    val_ds = val_ds.map(
        lambda images, labels: (normalizer(images), labels),
        num_parallel_calls=autotune,
    ).prefetch(autotune)
    return train_ds, val_ds, summary
