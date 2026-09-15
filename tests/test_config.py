import pytest

from src.config import TrainingConfig


def test_default_training_config_is_valid():
    cfg = TrainingConfig()
    cfg.validate()
    assert cfg.image_size == (224, 224)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"image_height": 0},
        {"image_width": 0},
        {"channels": 1},
        {"validation_fraction": 0},
        {"validation_fraction": 1},
        {"batch_size": 0},
        {"epochs": 0},
        {"learning_rate": 0},
        {"early_stopping_patience": -1},
    ],
)
def test_invalid_training_config_rejected(kwargs):
    with pytest.raises(ValueError):
        TrainingConfig(**kwargs).validate()
