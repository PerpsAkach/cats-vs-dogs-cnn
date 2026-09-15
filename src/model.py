from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers

from .config import DEFAULT_CONFIG, TrainingConfig


def build_cnn(config: TrainingConfig | None = None):
    cfg = config or DEFAULT_CONFIG
    cfg.validate()
    input_shape = (cfg.image_height, cfg.image_width, cfg.channels)

    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="cats_dogs_cnn",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg.learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="roc_auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def build_augmentation():
    """Recovered geometric augmentation expressed with Keras preprocessing layers."""

    return keras.Sequential(
        [
            layers.RandomRotation(20.0 / 360.0, fill_mode="nearest"),
            layers.RandomTranslation(0.1, 0.1, fill_mode="nearest"),
            layers.RandomZoom(0.1, 0.1, fill_mode="nearest"),
            layers.RandomFlip("horizontal"),
        ],
        name="augmentation",
    )
