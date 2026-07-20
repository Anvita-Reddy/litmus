from data import load_adult
from model import train_model, score
from audit import proxy_detection, worst_group


def main():
    X, y = load_adult()

    print("\n=== Litmus Step 4: audit the honest model ===")

    auc, proxies = proxy_detection(X, protected_col="sex")
    print(f"\nProxy check on 'sex':")
    print(f"   Reconstruction AUC = {auc:.3f}   (0.5 = safe, 1.0 = fully encoded)")
    print(f"   Strongest proxies: {', '.join(proxies)}")
    if auc > 0.8:
        print("   VERDICT: removing 'sex' is cosmetic - other features encode it.")


    model, X_test, y_test = train_model(X, y)
    base_acc, _ = score(model, X_test, y_test, "\nHeadline (average) accuracy:")

    groups = worst_group(model, X_test, y_test, group_col="sex")
    print("\nAccuracy by 'sex' subgroup (encoded 0=Female, 1=Male):")
    for val, acc in groups.items():
        print(f"   sex={val}:   {acc:.3f}")
    worst = min(groups.values())
    print(f"\n   Gap hidden by the average: {base_acc - worst:+.3f}")
    print()


if __name__ == "__main__":
    main()