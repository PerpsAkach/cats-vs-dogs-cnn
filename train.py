from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.data import build_arrays, split_data
from src.model import build_augmentation, build_cnn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model-out", type=Path, default=Path("outputs/cats_dogs_cnn.keras"))
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    x, y, paths = build_arrays(args.data)
    x_train, x_val, y_train, y_val, _, _ = split_data(x, y, paths)
    augmentation = build_augmentation()

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(len(x_train), seed=42).batch(args.batch_size)
    train_ds = train_ds.map(lambda images, labels: (augmentation(images, training=True), labels)).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    model = build_cnn()
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(args.model_out, monitor="val_loss", save_best_only=True),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ],
    )
    print(model.evaluate(val_ds, return_dict=True))


if __name__ == "__main__":
    main()
