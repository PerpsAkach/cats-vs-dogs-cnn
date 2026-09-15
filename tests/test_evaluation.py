import numpy as np
import pytest

from src.evaluation import evaluate_binary_predictions


def test_evaluation_returns_metrics_matrix_and_report():
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.6, 0.9])

    metrics, matrix, report = evaluate_binary_predictions(y_true, probabilities)

    assert metrics.accuracy == pytest.approx(1.0)
    assert metrics.precision == pytest.approx(1.0)
    assert metrics.recall == pytest.approx(1.0)
    assert metrics.f1 == pytest.approx(1.0)
    assert metrics.roc_auc == pytest.approx(1.0)
    assert matrix.tolist() == [[2, 0], [0, 2]]
    assert "cat" in report and "dog" in report


def test_single_class_input_has_no_roc_auc():
    metrics, matrix, _ = evaluate_binary_predictions([0, 0], [0.1, 0.2])
    assert metrics.roc_auc is None
    assert matrix.shape == (2, 2)


@pytest.mark.parametrize(
    "y_true, probabilities, threshold",
    [
        ([], [], 0.5),
        ([0, 1], [0.1], 0.5),
        ([0, 2], [0.1, 0.9], 0.5),
        ([0, 1], [0.1, np.nan], 0.5),
        ([0, 1], [0.1, 1.2], 0.5),
        ([0, 1], [0.1, 0.9], 0.0),
    ],
)
def test_invalid_evaluation_inputs_rejected(y_true, probabilities, threshold):
    with pytest.raises(ValueError):
        evaluate_binary_predictions(y_true, probabilities, threshold=threshold)
