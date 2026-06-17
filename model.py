"""Model training and evaluation."""

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

RANDOM_STATE = 37


def train_model(X, y):
    """Train XGBoost on an 80/20 split. Returns (model, X_test, y_test)."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    model = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.1,
        eval_metric="logloss", random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model, X_test, y_test


def score(model, X_test, y_test, label=""):
    """Print (if labelled) and return (accuracy, auc)."""
    acc = accuracy_score(y_test, model.predict(X_test))
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    if label:
        print(f"{label:<34} accuracy={acc:.3f}   AUC={auc:.3f}")
    return acc, auc


def train_and_score(X, y, label=""):
    """Train, score, return the model."""
    model, X_test, y_test = train_model(X, y)
    score(model, X_test, y_test, label)
    return model