import numpy as np

from src.model import build_cnn


def test_model_output_shape():
    model = build_cnn()
    x = np.zeros((2, 224, 224, 3), dtype=np.float32)
    y = model(x, training=False).numpy()
    assert y.shape == (2, 1)


def test_model_probability_range():
    model = build_cnn()
    x = np.zeros((1, 224, 224, 3), dtype=np.float32)
    p = float(model(x, training=False).numpy()[0, 0])
    assert 0 <= p <= 1
