from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from src.config import DEFAULT_CONFIG
from src.data import build_tf_datasets, discover_images, validate_images
from src.evaluation import evaluate_binary_predictions
from src.model import build_augmentation, build_cnn


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the reconstructed Cats vs Dogs CNN using streaming image datasets "
            "and export validation metrics and diagnostics."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG.epochs)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG.batch_size)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_CONFIG.learning_rate)
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=DEFAULT_CONFIG.validation_fraction,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_CONFIG.output_dir)
    parser.add_argument(
        "--verify-images",
        action="store_true",
        help="Verify every discovered image before training and fail on unreadable files.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = replace(
        DEFAULT_CONFIG,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_fraction=args.validation_fraction,
        output_dir=args.output_dir,
    )
    cfg.validate()

    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    tf.random.set_seed(cfg.random_seed)

    if args.verify_images:
        paths, _ = discover_images(args.data)
        unreadable = validate_images(paths)
        if unreadable:
            preview = ", ".join(str(path) for path in unreadable[:5])
            raise SystemExit(
                f"Unreadable images detected ({len(unreadable)}). First files: {preview}"
            )

    train_ds, val_ds, dataset_summary = build_tf_datasets(args.data, cfg)
    augmentation = build_augmentation()
    train_ds = train_ds.map(
        lambda images, labels: (augmentation(images, training=True), labels),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE)

    model = build_cnn(cfg)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = cfg.output_dir / "cats_dogs_cnn.keras"

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                model_path,
                monitor="val_loss",
                save_best_only=True,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=cfg.early_stopping_patience,
                restore_best_weights=True,
            ),
        ],
    )

    keras_metrics = model.evaluate(val_ds, return_dict=True, verbose=0)
    probabilities = model.predict(val_ds, verbose=0).reshape(-1)
    y_true = np.concatenate([labels.numpy().reshape(-1) for _, labels in val_ds])
    metrics, matrix, report = evaluate_binary_predictions(y_true, probabilities)

    pd.DataFrame(history.history).to_csv(cfg.output_dir / "training_history.csv", index=False)
    pd.DataFrame(
        matrix,
        index=["actual_cat", "actual_dog"],
        columns=["predicted_cat", "predicted_dog"],
    ).to_csv(cfg.output_dir / "confusion_matrix.csv")

    with (cfg.output_dir / "validation_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "binary_metrics": metrics.to_dict(),
                "keras_metrics": {k: float(v) for k, v in keras_metrics.items()},
            },
            handle,
            indent=2,
        )
    with (cfg.output_dir / "classification_report.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(report, handle, indent=2)
    with (cfg.output_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "configuration": {
                    **asdict(cfg),
                    "output_dir": str(cfg.output_dir),
                },
                "dataset": {
                    "root": str(dataset_summary.root),
                    "total_images": dataset_summary.total_images,
                    "cat_images": dataset_summary.cat_images,
                    "dog_images": dataset_summary.dog_images,
                    "class_balance_ratio": dataset_summary.class_balance_ratio,
                },
                "model_path": str(model_path),
            },
            handle,
            indent=2,
        )

    print(json.dumps(metrics.to_dict(), indent=2))


if __name__ == "__main__":
    main()
