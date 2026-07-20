from market_data import load_market
from audit import temporal_leakage, predictive_multiplicity


def main():
    X, y = load_market("SPY")
    print(f"{len(X)} days, {X.shape[1]} features, up-rate {y.mean():.1%}")

    r1, w1 = temporal_leakage(X, y)
    print(f"\nhorizon=1  (clean):      random {r1:.3f}  walk-forward {w1:.3f}  gap {r1 - w1:+.3f}")

    Xo, yo = load_market("SPY", horizon=20)
    r2, w2 = temporal_leakage(Xo, yo)
    print(f"horizon=20 (overlapping): random {r2:.3f}  walk-forward {w2:.3f}  gap {r2 - w2:+.3f}")

    r = predictive_multiplicity(X, y, n_models=25)
    frac = r["frac_pos"]
    substantial = ((frac >= 0.2) & (frac <= 0.8)).mean()
    print(f"\naccuracy range: {r['acc_min']:.3f} - {r['acc_max']:.3f}")
    print(f"substantial disagreement: {substantial * 100:.1f}%")


if __name__ == "__main__":
    main()