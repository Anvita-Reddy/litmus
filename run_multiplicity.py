"""Predictive multiplicity: how often equally-accurate models disagree per example."""

from data import load_adult
from audit import predictive_multiplicity


def main():
    X, y = load_adult()
    print("training 25 models...")
    r = predictive_multiplicity(X, y, n_models=25)
    frac = r["frac_pos"]

    print(f"accuracy range: {r['acc_min']:.3f} - {r['acc_max']:.3f}")

    any_disagree = ((frac > 0) & (frac < 1)).mean()
    substantial = (frac >= 0.2) & (frac <= 0.8)
    print(f"any disagreement:         {any_disagree * 100:.1f}%")
    print(f"substantial disagreement: {substantial.mean() * 100:.1f}%")

    X_test = r["X_test"]
    print("\nby sex:")
    for val in sorted(X_test["sex"].unique()):
        mask = (X_test["sex"] == val).values
        print(f"  sex={val}: {substantial[mask].mean() * 100:.1f}%")


if __name__ == "__main__":
    main()