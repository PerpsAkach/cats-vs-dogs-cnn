# Cats vs Dogs CNN

TensorFlow/Keras convolutional neural network for binary cat-vs-dog image classification using a 20,000-image dataset, augmentation, validation, and generalization analysis.

## Recovered experiment design

- 10,000 cat images
- 10,000 dog images
- 224×224 RGB preprocessing
- 80/20 train-validation split
- Conv2D blocks with 32 / 64 / 128 filters
- Dense(128) + Dropout(0.5)
- Adam optimizer, learning rate 0.001
- Binary cross-entropy
- Batch size 32
- 10 epochs

## Recovered final metrics

| Metric | Training | Validation |
|---|---:|---:|
| Accuracy | 97.23% | 82.30% |
| Loss | 0.0760 | 0.6565 |

The train-validation gap is documented as evidence of overfitting rather than presenting 97% as validated model performance.

## Run

```bash
pip install -r requirements.txt
python train.py --data path/to/train
```

See `PROVENANCE.md` for recovered vs reconstructed implementation details.
