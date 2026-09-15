# Output Contract

A successful training run writes artifacts under `--output-dir` (default `outputs/`).

## `cats_dogs_cnn.keras`

Best checkpoint by validation loss. The file is a Keras model artifact produced by the current reconstructed implementation.

## `training_history.csv`

One row per completed epoch with the metrics emitted by Keras, including training and validation loss/accuracy and any configured metrics.

## `validation_metrics.json`

Contains two sections:

- `binary_metrics`: thresholded validation metrics calculated from predicted probabilities (`accuracy`, `precision`, `recall`, `f1`, `roc_auc`, `threshold`, `observations`).
- `keras_metrics`: metrics returned directly by `model.evaluate`.

ROC-AUC is `null` when only one class is present in the evaluated labels.

## `confusion_matrix.csv`

Fixed 2 × 2 matrix with rows:

- `actual_cat`
- `actual_dog`

and columns:

- `predicted_cat`
- `predicted_dog`

The default decision threshold is `0.5`.

## `classification_report.json`

Per-class and aggregate precision, recall, F1-score, and support from scikit-learn.

## `run_manifest.json`

Records the current runtime configuration, dataset counts, class-balance ratio, dataset root supplied to the run, and checkpoint path.

## Interpretation boundary

Current run artifacts describe the dataset and execution that produced them. They do not retroactively validate or reproduce the historical 97.23% training accuracy / 82.30% validation accuracy unless a controlled rerun on the original dataset and configuration demonstrates that result.
