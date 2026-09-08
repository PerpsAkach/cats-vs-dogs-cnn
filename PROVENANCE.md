# Provenance

- **RECOVERED:** 20,000-image dataset, 224×224 RGB preprocessing, 80/20 split, 32/64/128 Conv2D architecture, Dense(128), Dropout(0.5), Adam 0.001, binary cross-entropy, augmentation settings, 10 epochs, batch size 32.
- **RECOVERED RESULTS:** 97.23% training accuracy, 82.30% validation accuracy, 0.0760 training loss, 0.6565 validation loss.
- **RECONSTRUCTED:** current TensorFlow/Keras code.
- **ENHANCED:** reusable modules, AUC/reporting, early stopping/checkpointing, and tests.
