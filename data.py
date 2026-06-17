import pandas as pd
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo


def load_adult():
    adult = fetch_ucirepo(name="Adult")
    X = adult.data.features.copy()

    y = adult.data.targets.iloc[:, 0].astype(str)
    y = y.str.replace(".", "", regex=False).str.strip()
    y = (y == ">50K").astype(int)

    X = X.replace("?", pd.NA)
    keep = X.notna().all(axis=1)
    X, y = X[keep].reset_index(drop=True), y[keep].reset_index(drop=True)

    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    for col in cat_cols:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    return X, y