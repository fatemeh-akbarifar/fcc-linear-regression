# Engineering notes

## Original project

Health-cost prediction with a dense neural network by Fatemeh Akbarifar, developed using a freeCodeCamp starter. The existing Git history and original Colab source establish the project provenance.

## Reproducibility work (September 2026)

- Replace notebook-only shell/magic commands in Python entry points with explicit download helpers and command-line interfaces.
- Pin dependencies and add focused tests plus a GitHub Actions workflow.
- Make imports free of downloads and training side effects.
- Save measured outputs separately from source code.
- Provide a notebook entry point that runs the same Python implementation.

## Method-specific changes

1. Use an 80/20 outer split with `random_state=0`; reserve 20% of the training pool for validation using seed 42.
2. Encode the original six features (`age`, `sex`, `bmi`, `children`, `smoker`, `region`) with fixed category mappings.
3. Fit a Keras normalization layer **only on training rows**, and include it in the saved model.
4. Train two 64-unit ReLU layers with RMSprop and MAE loss. Scale the output by 10,000 to improve optimization while retaining expense-unit predictions.
5. Restore the best validation-loss weights, then evaluate the held-out test set once. Report MAE and RMSE, and compare MAE with the challenge threshold of 3,500.

The modernization uses the simpler two-layer architecture from the later Colab version. It changes the original MSE training objective to MAE and adds early stopping. Ordinal region codes preserve the original approach but impose artificial ordering; one-hot encoding is a useful future comparison.

## Reading the evidence

The validation report describes newly executed runs. It does not retroactively claim that historical notebook outputs used the corrected evaluation pipeline. Unit tests verify behavior; they are not model-quality benchmarks.
