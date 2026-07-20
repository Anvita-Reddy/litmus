"""Feature-reliance stress test on a model with a planted shortcut."""

from data import load_adult
from model import train_model, score
from shortcuts import inject_shortcut
from audit import feature_reliance


def main():
    X, y = load_adult()
    model, X_test, y_test = train_model(inject_shortcut(X, y), y)
    base, _ = score(model, X_test, y_test, "with shortcut")

    reliance = feature_reliance(model, X_test, y_test)
    print("\nfeature reliance (accuracy drop when permuted):")
    for feat, drop in list(reliance.items())[:6]:
        print(f"  {feat:<18} {drop:.3f}")

    top = next(iter(reliance))
    print(f"\n{top} permuted: {base:.3f} -> {base - reliance[top]:.3f}")


if __name__ == "__main__":
    main()