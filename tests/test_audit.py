import numpy as np
import pandas as pd
import pytest

from model import train_model
from shortcuts import inject_shortcut
from audit import (
    feature_reliance,
    proxy_detection,
    worst_group,
    predictive_multiplicity,
    temporal_leakage,
)

RNG = np.random.default_rng(0)


@pytest.fixture(scope="module")
def synth():
    # two informative features, one noise feature, binary group column
    n = 3000
    x1 = RNG.normal(size=n)
    x2 = RNG.normal(size=n)
    noise = RNG.normal(size=n)
    group = RNG.integers(0, 2, size=n)
    y = pd.Series(((x1 + x2 + RNG.normal(scale=0.5, size=n)) > 0).astype(int))
    X = pd.DataFrame({"x1": x1, "x2": x2, "noise": noise, "group": group})
    return X, y


def test_feature_reliance_ranks_planted_shortcut_first(synth):
    X, y = synth
    model, X_test, y_test = train_model(inject_shortcut(X, y), y)
    reliance = feature_reliance(model, X_test, y_test)
    assert next(iter(reliance)) == "shortcut_feature"


def test_feature_reliance_ignores_noise(synth):
    X, y = synth
    model, X_test, y_test = train_model(X, y)
    reliance = feature_reliance(model, X_test, y_test)
    assert reliance["noise"] < 0.02


def test_proxy_detection_finds_planted_proxy(synth):
    X, y = synth
    X = X.copy()
    X["proxy"] = X["group"] * 2 + RNG.normal(scale=0.1, size=len(X))
    auc, top = proxy_detection(X, "group")
    assert auc > 0.95
    assert top[0] == "proxy"


def test_proxy_detection_low_when_independent(synth):
    X, y = synth
    auc, _ = proxy_detection(X[["x1", "x2", "noise", "group"]], "group")
    assert auc < 0.6


def test_worst_group_skips_tiny_groups(synth):
    X, y = synth
    X = X.copy()
    X.loc[X.index[:10], "group"] = 99  
    model, X_test, y_test = train_model(X, y)
    groups = worst_group(model, X_test, y_test, "group")
    assert 99 not in groups


def test_multiplicity_output_shape(synth):
    X, y = synth
    r = predictive_multiplicity(X, y, n_models=5)
    assert len(r["frac_pos"]) == len(r["X_test"])
    assert r["acc_min"] <= r["acc_max"]
    assert np.all((r["frac_pos"] >= 0) & (r["frac_pos"] <= 1))


def test_temporal_leakage_fires_on_overlapping_labels():
    n = 2000
    steps = RNG.normal(size=n).cumsum()
    df = pd.DataFrame({"level": steps})
    df["ret"] = df["level"].diff()
    df["ma"] = df["level"].rolling(10).mean() / df["level"]
    y = (df["level"].shift(-20) > df["level"]).astype(int)
    keep = df.notna().all(axis=1) & y.notna()
    X, y = df[keep].reset_index(drop=True), y[keep].reset_index(drop=True)

    rand_acc, wf_acc = temporal_leakage(X, y)
    assert rand_acc - wf_acc > 0.03
