from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers


def build_cnn(input_shape=(224, 224, 3), learning_rate=0.001):
    model = keras.Sequential([
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
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model


def build_augmentation():
    return keras.Sequential([
        layers.RandomRotation(20.0 / 360.0, fill_mode="nearest"),
        layers.RandomTranslation(0.1, 0.1, fill_mode="nearest"),
        layers.RandomZoom(0.1, 0.1, fill_mode="nearest"),
        layers.RandomFlip("horizontal"),
    ], name="augmentation")
