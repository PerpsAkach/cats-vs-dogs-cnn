# Provenance

## RECOVERED

Supported by prior project material and preserved historical records:

- 20,000-image binary dataset: 10,000 cats and 10,000 dogs.
- 224×224 RGB preprocessing.
- 80/20 train-validation split with random state 42.
- Three Conv2D stages with 32, 64, and 128 filters.
- Dense(128) and Dropout(0.5).
- Adam optimizer with learning rate 0.001.
- Binary cross-entropy loss.
- Batch size 32 and 10 epochs.
- Geometric augmentation corresponding to ±20° rotation, 0.1 width/height translation, 0.1 zoom, horizontal flipping, and nearest fill behavior.

### Recovered logged results

- Training accuracy: **97.23%**
- Validation accuracy: **82.30%**
- Training loss: **0.0760**
- Validation loss: **0.6565**

These values are historical logged results, not current CI benchmarks.

## RECONSTRUCTED

- Current TensorFlow/Keras CNN source.
- Current dataset-loading path.
- Current training CLI and output artifact structure.
- Current evaluation implementation.

The reconstruction is intended to preserve the supported experiment design without claiming that the literal original source bytes were recovered.

## ENHANCED

- Validated runtime configuration.
- Memory-efficient streaming via TensorFlow directory datasets instead of loading all images into one NumPy array.
- Explicit class ordering and dataset validation.
- Optional image-integrity verification.
- Precision, recall, F1, ROC-AUC, confusion matrix, and classification report.
- Training-history, validation-metrics, and run-manifest exports.
- Early stopping and best-validation-loss checkpointing.
- Automated tests, linting, dependency auditing, and CI runtime compatibility checks.
- Input/output contracts and implementation-status documentation.

## NOT CLAIMED / UNVERIFIED

- Literal original source code or commit history.
- Availability of the original 20,000 image files in this repository.
- Reproduction of the historical metrics by the current CI environment.
- State-of-the-art performance.
- Transfer-learning results not implemented here.
- Production deployment, model serving, robustness certification, or fairness certification.
