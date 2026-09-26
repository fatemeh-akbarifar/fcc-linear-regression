"""Predict insurance expenses with a dense neural-network regressor."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from data_utils import download

CATEGORIES = {"sex": {"male": 0, "female": 1}, "smoker": {"no": 0, "yes": 1},
              "region": {"southwest": 0, "southeast": 1, "northwest": 2, "northeast": 3}}
FEATURES = ["age", "sex", "bmi", "children", "smoker", "region"]


def encode_features(frame):
    result = frame.loc[:, FEATURES].copy()
    for column, mapping in CATEGORIES.items():
        if not result[column].isin(mapping).all():
            raise ValueError(f"Unknown or missing category in {column}")
        result[column] = result[column].map(mapping)
    result = result.astype("float32")
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("Features must be finite")
    return result


def build_model(training_features):
    normalizer = tf.keras.layers.Normalization()
    normalizer.adapt(np.asarray(training_features, dtype="float32"))
    inputs = tf.keras.Input(shape=(len(FEATURES),))
    x = normalizer(inputs)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dense(1)(x)
    # Scale the output, keeping the public API and metrics in expense units.
    outputs = tf.keras.layers.Rescaling(10000.0)(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001), loss="mae", metrics=["mae", "mse"])
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/insurance.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()
    data = pd.read_csv(download("https://cdn.freecodecamp.org/project-data/health-costs/insurance.csv", args.data))
    pool = data.sample(frac=0.8, random_state=0)
    test = data.drop(pool.index)
    train = pool.sample(frac=0.8, random_state=args.seed)
    validation = pool.drop(train.index)
    model = build_model(encode_features(train))
    history = model.fit(encode_features(train), train.expenses.to_numpy(),
                        validation_data=(encode_features(validation), validation.expenses.to_numpy()),
                        epochs=args.epochs, batch_size=32, verbose=2,
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True)])
    predictions = model.predict(encode_features(test), verbose=0).ravel()
    mae = float(np.mean(np.abs(test.expenses.to_numpy() - predictions)))
    metrics = {"mae": mae, "rmse": float(np.sqrt(np.mean((test.expenses.to_numpy() - predictions)**2))),
               "challenge_passed": mae < 3500, "train_rows": len(train), "validation_rows": len(validation),
               "test_rows": len(test), "epochs_run": len(history.history["loss"]), "seed": args.seed}
    args.output.mkdir(parents=True, exist_ok=True)
    model.save(args.output / "model.keras")
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (args.output / "history.json").write_text(json.dumps(history.history, indent=2))
    pd.DataFrame({"actual": test.expenses, "predicted": predictions}).to_csv(args.output / "predictions.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.scatter(test.expenses, predictions, s=12, alpha=.6)
    ax.plot([0, 65000], [0, 65000], "k--")
    ax.set(xlabel="Actual expenses", ylabel="Predicted expenses", title="Held-out health-cost predictions")
    fig.tight_layout(); fig.savefig(args.output / "predictions.png"); plt.close(fig)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
