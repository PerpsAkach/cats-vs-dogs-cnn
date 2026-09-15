# Input Contract

## Dataset layout

The training CLI expects a directory with exactly the two class subdirectories used by the reconstructed experiment:

```text
<dataset-root>/
├── cats/
│   └── ... image files ...
└── dogs/
    └── ... image files ...
```

Supported extensions are `.jpg`, `.jpeg`, `.png`, and `.bmp` (case-insensitive).

## Class semantics

- `cats` → label `0`
- `dogs` → label `1`

The TensorFlow loader is given this class order explicitly so class indices are not inferred from an accidental directory ordering.

## Image preprocessing

- Images are decoded as RGB.
- Images are resized to the configured `(height, width)`, default `224 × 224`.
- Pixel values are rescaled by `1 / 255` into `[0, 1]`.
- The default validation split is `20%` with random seed `42`.

## Dataset validation

The pipeline checks that:

- the dataset root exists and is a directory;
- both `cats/` and `dogs/` directories exist;
- both classes contain at least one supported image.

`--verify-images` additionally asks Pillow to verify every discovered file before training and stops if unreadable files are found. This scan is optional because it adds I/O cost.

## Memory behavior

The training path uses `tf.keras.utils.image_dataset_from_directory` and `tf.data` instead of materializing the entire image collection as one NumPy array. This avoids requiring the full dataset to fit in RAM.

## Historical-data boundary

The original image files are not committed to this portfolio repository. The documented historical experiment used 20,000 images (10,000 cats and 10,000 dogs), but a current run operates on whatever valid dataset the user supplies and must not be assumed to reproduce the historical metrics automatically.
