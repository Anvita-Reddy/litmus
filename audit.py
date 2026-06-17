"""Audit checks: what a model relies on, what it hides, and how arbitrary it is."""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from model import RANDOM_STATE


def feature_reliance(model, X_test, y_test):
    """Permutation importance as accuracy drop. Ranks features by how much the
    model leans on them -> the auto-discovery of shortcuts."""
    rng = np.random.default_rng(RANDOM_STATE)
    base_acc = accuracy_score(y_test, model.predict(X_test))
    drops = {}
    for col in X_test.columns:
        X_perm = X_test.copy()
        X_perm[col] = rng.permutation(X_perm[col].values)
        drops[col] = base_acc - accuracy_score(y_test, model.predict(X_perm))
    return dict(sorted(drops.items(), key=lambda kv: kv[1], reverse=True))


def proxy_detection(X, protected_col):
    """Can the protected attribute be rebuilt from the OTHER features?
    Returns (auc, top_3_proxies). AUC ~0.5 = truly removed; ~1.0 = still encoded."""
    target = X[protected_col]
    if target.nunique() > 2:
        target = pd.factorize(target)[0]
    X_rest = X.drop(columns=[protected_col])
    X_tr, X_te, t_tr, t_te = train_test_split(
        X_rest, target, test_size=0.2, random_state=RANDOM_STATE, stratify=target
    )
    probe = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.1,
                          eval_metric="logloss", random_state=RANDOM_STATE)
    probe.fit(X_tr, t_tr)
    if len(np.unique(target)) == 2:
        auc = roc_auc_score(t_te, probe.predict_proba(X_te)[:, 1])
    else:
        auc = roc_auc_score(t_te, probe.predict_proba(X_te), multi_class="ovr")
    importances = dict(zip(X_rest.columns, probe.feature_importances_))
    top_proxies = sorted(importances, key=importances.get, reverse=True)[:3]
    return auc, top_proxies


def worst_group(model, X_test, y_test, group_col):
    """Per-group accuracy, sorted worst -> best. Shows what the average hides."""
    preds = model.predict(X_test)
    out = {}
    for val in sorted(X_test[group_col].unique()):
        mask = X_test[group_col] == val
        if mask.sum() < 30:
            continue
        out[val] = accuracy_score(y_test[mask], preds[mask])
    return dict(sorted(out.items(), key=lambda kv: kv[1]))


def predictive_multiplicity(X, y, n_models=25, subsample=0.9):
    """Train many EQUALLY-ACCURATE models and measure how often they disagree
    about the SAME individual.

    Each model = a different random seed + a different 90% subsample of the
    training rows. These are tiny perturbations that leave overall accuracy
    basically unchanged. But if two equally-accurate models give one person
    opposite decisions, that person's outcome was decided by chance, not data.

    Returns accuracy spread + the per-row fraction of models predicting class 1.
    """
    X_tr_full, X_test, y_tr_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    preds = np.zeros((n_models, len(X_test)), dtype=int)
    accs = []
    for i in range(n_models):
        seed = RANDOM_STATE + i
        Xi = X_tr_full.sample(frac=subsample, random_state=seed)
        yi = y_tr_full.loc[Xi.index]
        m = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                          eval_metric="logloss", random_state=seed)
        m.fit(Xi, yi)
        preds[i] = m.predict(X_test)
        accs.append(accuracy_score(y_test, preds[i]))

    frac_pos = preds.mean(axis=0)  # share of models predicting "yes" per person
    return {
        "accs": accs,
        "acc_min": float(np.min(accs)),
        "acc_max": float(np.max(accs)),
        "frac_pos": frac_pos,
        "X_test": X_test,
        "y_test": y_test,
    }