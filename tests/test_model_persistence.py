"""Small archive tests independent of expensive stage-module execution."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from tools.model_persistence import load_model_archive, predict_state, save_model_archive


def test_regressor_roundtrip_and_manifest(tmp_path):
    X = pd.DataFrame({"second": np.arange(12), "first": np.arange(12) ** 2})
    model = RandomForestRegressor(n_estimators=3, random_state=42).fit(X, X["first"])
    state = {"kind": "regressors", "models": {"y_avg": model}}
    expected = {"pred_avg": model.predict(X)}
    folder = save_model_archive(tmp_path, "rf", state, X.columns, X, expected)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["features"] == ["second", "first"]
    assert manifest["verified_rows"] == len(X)
    assert manifest["packages"]["numpy"] == np.__version__
    assert manifest["sha256"]["model.joblib"] == hashlib.sha256((folder / "model.joblib").read_bytes()).hexdigest()
    np.testing.assert_array_equal(predict_state(load_model_archive(folder)["state"], X)["pred_avg"], expected["pred_avg"])


def test_ensemble_recipe_roundtrip(tmp_path):
    X = pd.DataFrame({"y_avg_lag168": [2., 4.], "y_peak_lag168": [6., 8.]})
    state = {"kind": "ensemble", "weight": .3, "names": ["a", "b"],
             "members": [{"kind": "naive"}, {"kind": "naive"}]}
    folder = save_model_archive(tmp_path, "ensemble", state, X.columns, X, predict_state(state, X))
    assert load_model_archive(folder)["state"]["names"] == ["a", "b"]


def test_reject_changed_predictions(tmp_path):
    X = pd.DataFrame({"y_avg_lag168": [2.], "y_peak_lag168": [6.]})
    with pytest.raises(AssertionError):
        save_model_archive(tmp_path, "bad", {"kind": "naive"}, X.columns, X, {"pred_avg": [99.]})


def test_keras_and_scalers_roundtrip(tmp_path):
    tf = pytest.importorskip("tensorflow")
    from sklearn.preprocessing import StandardScaler
    X = pd.DataFrame({"value": [1., 2., 3., 4.]})
    sc = StandardScaler().fit(X)
    model = tf.keras.Sequential([tf.keras.layers.Input((1,)), tf.keras.layers.Dense(1)])
    state = {"kind": "keras", "models": {"y_avg": model}, "xsc": sc, "yscalers": {"y_avg": sc}}
    folder = save_model_archive(tmp_path, "dnn", state, X.columns, X, predict_state(state, X))
    assert (folder / "network_0.keras").is_file()


def test_original_rnn_joint_scaler_roundtrip(tmp_path):
    tf = pytest.importorskip("tensorflow")
    from sklearn.preprocessing import MinMaxScaler
    mat = np.random.default_rng(42).normal(size=(5, 169))
    scaler = MinMaxScaler().fit(mat)
    model = tf.keras.Sequential([
        tf.keras.layers.Input((24, 7)), tf.keras.layers.SimpleRNN(2), tf.keras.layers.Dense(1),
    ])
    raw = model.predict(scaler.transform(mat)[:, 1:].reshape(-1, 24, 7), verbose=0).ravel()
    expected = raw * scaler.data_range_[0] + scaler.data_min_[0]
    state = {"kind": "rnn_original", "model": model, "scaler": scaler}
    save_model_archive(tmp_path, "rnn", state, list(range(168)), mat[:, 1:], {"pred_avg": expected})


def test_regime_routing_and_fallback(tmp_path):
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    X = pd.DataFrame({"gate": [0., 0., 1., 1.], "value": [1., 2., 3., 4.]})
    gate = DecisionTreeClassifier().fit(X[["gate"]], [0, 0, 1, 1])
    specialist = DecisionTreeRegressor().fit(X.iloc[:2], [10., 20.])
    fallback = DecisionTreeRegressor().fit(X, [30., 40., 50., 60.])
    state = {"kind": "regime", "gate": gate, "gcols": ["gate"],
             "regs": {t: {0: specialist} for t in ("y_avg", "y_peak")},
             "fallbacks": {t: fallback for t in ("y_avg", "y_peak")}}
    expected = {"pred_avg": [10., 20., 50., 60.], "pred_peak": [10., 20., 50., 60.]}
    save_model_archive(tmp_path, "regime", state, X.columns, X, expected)
