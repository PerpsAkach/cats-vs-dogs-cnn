from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class BinaryMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    threshold: float
    observations: int

    def to_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


def evaluate_binary_predictions(
    y_true,
    probabilities,
    threshold: float = 0.5,
) -> tuple[BinaryMetrics, np.ndarray, dict]:
    y_true_arr = np.asarray(y_true).reshape(-1)
    prob_arr = np.asarray(probabilities, dtype=float).reshape(-1)

    if y_true_arr.size == 0:
        raise ValueError("y_true must contain at least one observation")
    if y_true_arr.shape != prob_arr.shape:
        raise ValueError("y_true and probabilities must have the same length")
    if not 0 < threshold < 1:
        raise ValueError("threshold must be between 0 and 1")
    if not np.all(np.isfinite(prob_arr)):
        raise ValueError("probabilities must be finite")
    if np.any((prob_arr < 0) | (prob_arr > 1)):
        raise ValueError("probabilities must be in [0, 1]")

    labels = set(np.unique(y_true_arr).tolist())
    if not labels.issubset({0, 1}):
        raise ValueError("y_true must contain only binary labels 0 and 1")

    predicted = (prob_arr >= threshold).astype(int)
    roc_auc = None
    if len(np.unique(y_true_arr)) == 2:
        roc_auc = float(roc_auc_score(y_true_arr, prob_arr))

    metrics = BinaryMetrics(
        accuracy=float(accuracy_score(y_true_arr, predicted)),
        precision=float(precision_score(y_true_arr, predicted, zero_division=0)),
        recall=float(recall_score(y_true_arr, predicted, zero_division=0)),
        f1=float(f1_score(y_true_arr, predicted, zero_division=0)),
        roc_auc=roc_auc,
        threshold=float(threshold),
        observations=int(y_true_arr.size),
    )
    matrix = confusion_matrix(y_true_arr, predicted, labels=[0, 1])
    report = classification_report(
        y_true_arr,
        predicted,
        labels=[0, 1],
        target_names=["cat", "dog"],
        output_dict=True,
        zero_division=0,
    )
    return metrics, matrix, report
