from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.data import discover_images, load_image, summarize_dataset, validate_images


def _write_rgb(path: Path, value: int = 128) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(np.full((8, 8, 3), value, dtype=np.uint8), mode="RGB")
    image.save(path)


def test_discovery_and_summary(tmp_path):
    _write_rgb(tmp_path / "cats" / "cat1.jpg", 32)
    _write_rgb(tmp_path / "cats" / "cat2.png", 64)
    _write_rgb(tmp_path / "dogs" / "dog1.jpg", 192)
    (tmp_path / "dogs" / "ignore.txt").write_text("not an image", encoding="utf-8")

    paths, labels = discover_images(tmp_path)
    summary = summarize_dataset(tmp_path)

    assert len(paths) == 3
    assert labels.count(0) == 2
    assert labels.count(1) == 1
    assert summary.total_images == 3
    assert summary.cat_images == 2
    assert summary.dog_images == 1
    assert summary.class_balance_ratio == pytest.approx(0.5)


def test_load_image_returns_normalized_rgb(tmp_path):
    path = tmp_path / "image.jpg"
    _write_rgb(path, 255)
    image = load_image(path, image_size=(4, 5))
    assert image.shape == (4, 5, 3)
    assert image.dtype == np.float32
    assert image.min() >= 0
    assert image.max() <= 1


def test_missing_class_directory_rejected(tmp_path):
    (tmp_path / "cats").mkdir()
    with pytest.raises(FileNotFoundError, match="dogs"):
        discover_images(tmp_path)


def test_summary_requires_both_classes(tmp_path):
    _write_rgb(tmp_path / "cats" / "cat.jpg")
    (tmp_path / "dogs").mkdir()
    with pytest.raises(ValueError, match="Both cats and dogs"):
        summarize_dataset(tmp_path)


def test_validate_images_detects_corrupt_file(tmp_path):
    good = tmp_path / "good.jpg"
    bad = tmp_path / "bad.jpg"
    _write_rgb(good)
    bad.write_bytes(b"not really an image")
    assert validate_images([good, bad]) == [bad]
