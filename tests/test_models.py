"""Smoke tests for the model pipelines in pyrregular.models.

Each pipeline is cloned (the module-level objects are shared), fitted briefly on
a small synthetic irregular dataset, and must predict valid labels. A test is
skipped when the libraries its pipeline needs are not installed, so this file
also runs (all skipped) in an environment without the models extra.
"""

import importlib

import numpy as np
import pytest
from sklearn.base import clone

pytestmark = pytest.mark.models

N_SERIES, N_SIGNALS, MAX_LEN = 40, 3, 30


@pytest.fixture(scope="module")
def data():
    """Padded irregular series as to_dense returns them, two classes."""
    rng = np.random.default_rng(0)
    X = np.full((N_SERIES, N_SIGNALS, MAX_LEN), np.nan)
    T = np.full((N_SERIES, 1, MAX_LEN), np.nan)
    y = np.arange(N_SERIES) % 2
    for i in range(N_SERIES):
        n = rng.integers(15, MAX_LEN + 1)
        times = np.sort(rng.choice(200, size=n, replace=False)).astype(float)
        T[i, 0, :n] = times
        X[i, :, :n] = np.sin(times / 10 + 2 * y[i]) + 0.1 * rng.normal(
            size=(N_SIGNALS, n)
        )
        X[i, :, :n][rng.random((N_SIGNALS, n)) < 0.2] = np.nan  # missing values
    train = np.arange(N_SERIES) % 4 != 0
    return X, np.concatenate([X, T], axis=1), y, train


# module, pipeline name, libraries it needs, whether it needs the time channel
PIPELINES = [
    ("borf", "borf_pipeline", ["aeon", "lightgbm"], False),
    ("brits", "brits_pipeline", ["pypots"], False),
    ("grud", "grud_pipeline", ["pypots"], False),
    ("knn", "knn_dtw", ["tslearn"], False),
    ("lgbm", "lgbm_pipeline", ["sktime", "lightgbm"], False),
    ("ncde", "ncde_pipeline", ["jax", "equinox", "optax", "diffrax"], True),
    ("raindrop", "raindrop_pipeline", ["pypots", "torch_scatter"], False),
    ("rifc", "rifc_pipeline", ["sktime", "lightgbm"], False),
    ("rocket", "rocket_pipeline", ["sktime", "lightgbm"], False),
    ("saits", "saits_pipeline", ["pypots"], False),
    ("svm", "svm_pipeline", ["sktime", "tslearn"], False),
    ("timesnet", "timesnet_pipeline", ["pypots"], False),
]


def _short(model):
    """Train deep models for as little as possible."""
    if hasattr(model, "model_params"):  # pypots wrappers
        model.model_params = {**model.model_params, "epochs": 1, "patience": 1}
    if hasattr(model, "max_iter") and hasattr(model, "print_step"):  # NCDE
        model.set_params(max_iter=2, print_step=1000)
    return model


def _pipeline(module, name, libs):
    for lib in libs:
        pytest.importorskip(lib)
    return getattr(importlib.import_module(f"pyrregular.models.{module}"), name)


@pytest.mark.parametrize(
    "module, name, libs, needs_time", PIPELINES, ids=[p[0] for p in PIPELINES]
)
def test_pipeline_fit_predict(data, module, name, libs, needs_time):
    X, X_time, y, train = data
    X = X_time if needs_time else X
    model = _short(clone(_pipeline(module, name, libs)))
    model.fit(X[train], y[train])
    pred = np.asarray(model.predict(X[~train]))
    assert pred.shape == ((~train).sum(),)
    assert set(np.unique(pred)) <= set(np.unique(y))
    proba = np.asarray(model.predict_proba(X[~train]))
    assert proba.shape == ((~train).sum(), 2)


def test_pypots_any_labels_and_refit(data):
    X, _, y, train = data
    model = _short(clone(_pipeline("grud", "grud_pipeline", ["pypots"])))
    labels = np.array(["run", "walk"])[y]
    model.fit(X[train], labels[train])
    model.fit(X[train], labels[train])  # a second fit used to fail
    assert set(model.predict(X[~train])) <= {"run", "walk"}
    assert list(model.classes_) == ["run", "walk"]


def test_rifc_supports_sklearn_tools(data):
    from sklearn.model_selection import cross_val_score

    X, _, y, _ = data
    rifc = _pipeline("rifc", "rifc_pipeline", ["sktime", "lightgbm"])
    assert "randomintervalfeatureclassifier__n_intervals" in rifc.get_params()
    scores = cross_val_score(clone(rifc), X, y, cv=2)
    assert scores.shape == (2,)
