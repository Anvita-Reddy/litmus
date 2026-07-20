import numpy as np
import pandas as pd

from market_data import load_market


def _sharpe(returns):
    r = np.asarray(returns, dtype=float)
    if r.std() == 0 or len(r) == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(252))


def random_strategy_search(n_strategies=200, seed=0):
    X, y = load_market("SPY")
    ret = X["ret1"].shift(-1).fillna(0.0).values
    feats = X.values
    n = len(X)
    split = int(n * 0.6)

    rng = np.random.default_rng(seed)
    results = []
    for _ in range(n_strategies):
        k = rng.integers(1, feats.shape[1] + 1)
        cols = rng.choice(feats.shape[1], size=k, replace=False)
        signs = rng.choice([-1, 1], size=k)
        score = (feats[:, cols] * signs).sum(axis=1)
        position = (score > np.median(score[:split])).astype(float)
        strat_ret = position * ret
        results.append({
            "in_sample": _sharpe(strat_ret[:split]),
            "out_sample": _sharpe(strat_ret[split:]),
        })

    df = pd.DataFrame(results)
    best = df.loc[df["in_sample"].idxmax()]
    return {
        "n_strategies": int(n_strategies),
        "best_in_sample": float(best["in_sample"]),
        "best_out_sample": float(best["out_sample"]),
        "in_sample_all": [round(float(v), 3) for v in df["in_sample"]],
        "out_sample_all": [round(float(v), 3) for v in df["out_sample"]],
        "median_in_sample": float(df["in_sample"].median()),
    }


if __name__ == "__main__":
    r = random_strategy_search()
    print(f"tested {r['n_strategies']} random strategies")
    print(f"best in-sample Sharpe:  {r['best_in_sample']:.2f}")
    print(f"its out-of-sample Sharpe: {r['best_out_sample']:.2f}")
