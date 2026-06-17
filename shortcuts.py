import numpy as np

from model import RANDOM_STATE


def inject_shortcut(X, y, leak_rate=0.05, random_state=RANDOM_STATE):
    """Add a label-correlated feature (matches the label on 1 - leak_rate of rows)."""
    rng = np.random.default_rng(random_state)
    flip = rng.random(len(y)) < leak_rate
    out = X.copy()
    out["shortcut_feature"] = np.where(flip, 1 - y, y)
    return out