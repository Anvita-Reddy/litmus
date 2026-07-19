import json

from data import load_adult
from market_data import load_market
from compas_data import load_compas
from model import train_model, score
from shortcuts import inject_shortcut
from audit import (
    feature_reliance,
    proxy_detection,
    worst_group,
    predictive_multiplicity,
    temporal_leakage,
)


def main():
    results = {}

    # --- adult ---
    X, y = load_adult()

    model, X_test, y_test = train_model(X, y)
    base_acc, base_auc = score(model, X_test, y_test)
    results["baseline"] = {"acc": base_acc, "auc": base_auc}

    cheat_model, Xc_test, yc_test = train_model(inject_shortcut(X, y), y)
    cheat_acc, cheat_auc = score(cheat_model, Xc_test, yc_test)
    reliance = feature_reliance(cheat_model, Xc_test, yc_test)
    top = next(iter(reliance))
    results["shortcut"] = {
        "acc": cheat_acc, "auc": cheat_auc, "reliance": reliance,
        "top_feature": top, "collapsed_acc": cheat_acc - reliance[top],
    }

    auc, proxies = proxy_detection(X, "sex")
    results["proxy"] = {"attribute": "sex", "auc": auc, "top_proxies": proxies}

    results["worst_group"] = {
        "by_sex": {str(k): v for k, v in worst_group(model, X_test, y_test, "sex").items()}
    }

    r = predictive_multiplicity(X, y, n_models=25)
    results["multiplicity_adult"] = {
        "acc_min": r["acc_min"], "acc_max": r["acc_max"],
        "frac_pos": [round(float(f), 3) for f in r["frac_pos"]],
    }

    # --- market ---
    Xm, ym = load_market("SPY")
    r1, w1 = temporal_leakage(Xm, ym)
    Xo, yo = load_market("SPY", horizon=20)
    r2, w2 = temporal_leakage(Xo, yo)
    results["leakage"] = {
        "clean": {"random": r1, "walkforward": w1},
        "overlapping": {"random": r2, "walkforward": w2},
    }

    rm = predictive_multiplicity(Xm, ym, n_models=25)
    results["multiplicity_market"] = {
        "acc_min": rm["acc_min"], "acc_max": rm["acc_max"],
        "frac_pos": [round(float(f), 3) for f in rm["frac_pos"]],
        "up_rate": float(ym.mean()),
    }

    # --- compas ---
    Xc, yc = load_compas()
    cmodel, Xc_test, yc_test = train_model(Xc, yc)
    c_acc, c_auc = score(cmodel, Xc_test, yc_test)

    c_proxy_auc, c_proxies = proxy_detection(Xc, "race")
    c_groups = worst_group(cmodel, Xc_test, yc_test, "race")
    rc = predictive_multiplicity(Xc, yc, n_models=25)

    results["compas"] = {
        "acc": c_acc, "auc": c_auc, "n": int(len(Xc)),
        "proxy": {"attribute": "race", "auc": c_proxy_auc, "top_proxies": c_proxies},
        "worst_group": {str(k): v for k, v in c_groups.items()},
        "multiplicity": {
            "acc_min": rc["acc_min"], "acc_max": rc["acc_max"],
            "frac_pos": [round(float(f), 3) for f in rc["frac_pos"]],
        },
    }

    with open("results.json", "w") as f:
        json.dump(results, f)
    print("wrote results.json")


if __name__ == "__main__":
    main()