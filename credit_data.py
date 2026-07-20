import pandas as pd
from ucimlrepo import fetch_ucirepo


def load_credit():
    ds = fetch_ucirepo(id=144)  # Statlog German Credit Data
    X_raw = ds.data.features.copy()
    y_raw = ds.data.targets.copy()

    y = (y_raw.iloc[:, 0] == 2).astype(int).reset_index(drop=True)

    status_col = None
    for c in X_raw.columns:
        vals = set(X_raw[c].astype(str).unique())
        if vals & {"A91", "A92", "A93", "A94", "A95"}:
            status_col = c
            break

    out = pd.DataFrame()
    if status_col is not None:
        out["sex"] = X_raw[status_col].astype(str).map(
            {"A91": 1, "A93": 1, "A94": 1, "A92": 0, "A95": 0}
        ).fillna(1).astype(int)

    for c in X_raw.columns:
        if pd.api.types.is_numeric_dtype(X_raw[c]):
            s = X_raw[c]
            if s.min() >= 18 and s.max() <= 80 and s.between(18, 80).mean() > 0.95:
                out["age"] = s.reset_index(drop=True)
                break
    if "age" in out:
        out["age_over_45"] = (out["age"] > 45).astype(int)


    for c in X_raw.columns:
        if c == status_col:
            continue
        col = X_raw[c].reset_index(drop=True)
        if pd.api.types.is_numeric_dtype(col):
            out[c] = col
        else:
            out[c] = col.astype("category").cat.codes

    out = out.loc[:, ~out.columns.duplicated()]
    return out.reset_index(drop=True), y