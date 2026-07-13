import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from model import RANDOM_STATE


def feature_reliance(model, X_test, y_test):
    rng = np.random.default_rng(RANDOM_STATE)
    base = accuracy_score(y_test, model.predict(X_test))
    drops = {}
    for col in X_test.columns:
        perturbed = X_test.copy()
        perturbed[col] = rng.permutation(perturbed[col].values)
        drops[col] = base - accuracy_score(y_test, model.predict(perturbed))
    return dict(sorted(drops.items(), key=lambda kv: kv[1], reverse=True))


def proxy_detection(X, protected_col):
    target = X[protected_col]
    if target.nunique() > 2:
        target = pd.factorize(target)[0]
    rest = X.drop(columns=[protected_col])

    X_tr, X_te, t_tr, t_te = train_test_split(
        rest, target, test_size=0.2, random_state=RANDOM_STATE, stratify=target
    )
    probe = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.1,
                          eval_metric="logloss", random_state=RANDOM_STATE)
    probe.fit(X_tr, t_tr)

    if len(np.unique(target)) == 2:
        auc = roc_auc_score(t_te, probe.predict_proba(X_te)[:, 1])
    else:
        auc = roc_auc_score(t_te, probe.predict_proba(X_te), multi_class="ovr")

    importance = dict(zip(rest.columns, probe.feature_importances_))
    top = sorted(importance, key=importance.get, reverse=True)[:3]
    return auc, top  # auc ~0.5 = not recoverable, ~1.0 = still encoded


def worst_group(model, X_test, y_test, group_col):
    preds = model.predict(X_test)
    out = {}
    for val in sorted(X_test[group_col].unique()):
        mask = X_test[group_col] == val
        if mask.sum() < 30:
            continue
        out[val] = accuracy_score(y_test[mask], preds[mask])
    return dict(sorted(out.items(), key=lambda kv: kv[1]))


def predictive_multiplicity(X, y, n_models=25, subsample=0.9):
    X_tr, X_test, y_tr, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    preds = np.zeros((n_models, len(X_test)), dtype=int)
    accs = []
    for i in range(n_models):
        seed = RANDOM_STATE + i
        Xi = X_tr.sample(frac=subsample, random_state=seed)
        yi = y_tr.loc[Xi.index]
        m = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                          eval_metric="logloss", random_state=seed)
        m.fit(Xi, yi)
        preds[i] = m.predict(X_test)
        accs.append(accuracy_score(y_test, preds[i]))

    return {
        "accs": accs,
        "acc_min": float(np.min(accs)),
        "acc_max": float(np.max(accs)),
        "frac_pos": preds.mean(axis=0),
        "X_test": X_test,
        "y_test": y_test,
    }


def temporal_leakage(X, y, test_frac=0.2):
    # X, y must be in chronological order
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
        X, y, test_size=test_frac, random_state=RANDOM_STATE, shuffle=True, stratify=y
    )
    rand_model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                               eval_metric="logloss", random_state=RANDOM_STATE)
    rand_model.fit(Xr_tr, yr_tr)
    random_acc = accuracy_score(yr_te, rand_model.predict(Xr_te))

    cut = int(len(X) * (1 - test_frac))
    wf_model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                             eval_metric="logloss", random_state=RANDOM_STATE)
    wf_model.fit(X.iloc[:cut], y.iloc[:cut])
    wf_acc = accuracy_score(y.iloc[cut:], wf_model.predict(X.iloc[cut:]))
    return random_acc, wf_acc