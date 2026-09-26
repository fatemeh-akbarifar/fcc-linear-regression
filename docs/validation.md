# Reproducibility report

Verified on 26 September 2026 with Python 3.9.13 on macOS ARM64 (CPU).

## Dataset run

```bash
python PredictHealthCosts.py
```

The validation run used the same public data cached under `/private/tmp/fcc-data`; output paths were supplied explicitly where needed. Defaults use repository-local `data/` and `artifacts/`.

Held-out MAE: **2,198.32 expense units**; RMSE: **5,618.40**. The 3,500 MAE challenge threshold was met. Training used 856 rows, validation 214 rows, and test 268 rows; early stopping ended after 159 epochs.

[Machine-readable results](results.json). These are newly measured results, not historical notebook output.

## Behavioral checks

`python -m pytest -q` passed 3 tests. Tests cover archive traversal rejection and invalid categories, normalization persistence, one training step, and prediction consistency after saving/loading.

## Environment

Core versions: NumPy 1.23.5, pandas 2.2.3, TensorFlow 2.16.2 and Keras 3.7.0. Dependency pins are in `requirements.txt`. No GPU was used. Results can vary across platforms; the neural-network seed is 42.

The GitHub Actions workflow is configured separately; local success does not itself establish a successful hosted workflow run.
