import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from PredictHealthCosts import encode_features, build_model


def features():
    return pd.DataFrame({"age": [20,40,60], "sex": ["male","female","male"], "bmi": [20,25,30],
                         "children": [0,1,2], "smoker": ["no","yes","no"], "region": ["southwest","northeast","southeast"]})


def test_unknown_category():
    frame = features(); frame.loc[0, "region"] = "unknown"
    with pytest.raises(ValueError): encode_features(frame)


def test_normalization_persists_for_raw_input(tmp_path):
    raw = encode_features(features()).to_numpy()
    model = build_model(raw)
    expected = model(raw, training=False).numpy()
    path = tmp_path / "model.keras"; model.save(path)
    loaded = tf.keras.models.load_model(path)
    np.testing.assert_allclose(loaded(raw, training=False).numpy(), expected, rtol=1e-5)
    normalizer = next(x for x in loaded.layers if isinstance(x, tf.keras.layers.Normalization))
    np.testing.assert_allclose(normalizer.mean.numpy().ravel(), raw.mean(axis=0))
    assert np.isfinite(model.train_on_batch(raw, np.array([1000.,2000.,3000.]))).all()
