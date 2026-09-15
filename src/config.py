from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    """Runtime configuration for the reconstructed Cats vs Dogs pipeline."""

    image_height: int = 224
    image_width: int = 224
    channels: int = 3
    validation_fraction: float = 0.20
    random_seed: int = 42
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 0.001
    early_stopping_patience: int = 3
    output_dir: Path = Path("outputs")

    def validate(self) -> None:
        if self.image_height <= 0 or self.image_width <= 0:
            raise ValueError("image dimensions must be positive")
        if self.channels != 3:
            raise ValueError("this pipeline expects 3-channel RGB images")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be non-negative")

    @property
    def image_size(self) -> tuple[int, int]:
        return self.image_height, self.image_width


DEFAULT_CONFIG = TrainingConfig()
DEFAULT_CONFIG.validate()
