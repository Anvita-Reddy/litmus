# Litmus

A model-auditing toolkit. High accuracy doesn't mean a model is trustworthy, and Litmus checks for the common reasons a good score can be misleading: a leaked feature doing the work, a protected attribute still encoded in the data after you drop it, large gaps between subgroups, and predictions that flip depending on the random seed.

Each check is tested against a planted problem first, so I know it actually detects what it claims to. Then it's run on real datasets: census income, stock prediction, criminal-risk scores, credit, and trading backtests.

Live dashboard: https://litmus-audit.streamlit.app

![tests](https://github.com/Anvita-Reddy/litmus/actions/workflows/test.yml/badge.svg)

## Checks

| Check | What it looks for |
|---|---|
| Shortcut detection | Accuracy resting on a leaked feature that won't be available at prediction time |
| Proxy detection | A dropped protected attribute still recoverable from the remaining features |
| Subgroup audit | A headline accuracy that hides large per-group gaps |
| Predictive multiplicity | Equally-accurate models that disagree on individual predictions |
| Temporal leakage | A backtest inflated by future information leaking across the split |

Every check has positive and negative controls in the test suite: it fires on a planted problem and stays quiet on clean data.

## Results on real data

Adult / Census
- Planting one leaked feature raises accuracy from 87.3% to 96.9%. The reliance ranking flags it without being told where to look. Permuting that feature drops accuracy to 76.7%, below the honest baseline.
- With `sex` dropped from the features, a probe model recovers it at 0.94 AUC.

SPY (stock direction)
- A shuffled backtest reports 65.9%. Walk-forward validation on the same setup reports 57.8%. The gap is leakage.
- Across 200 random strategies with no real edge, the best in-sample Sharpe is 1.11 and drops to 0.71 out-of-sample. The ranking is selection noise.

COMPAS (criminal risk)
- A model trained on the public COMPAS data (5,278 defendants) scores 68.0%. With `race` removed, a probe recovers it at 0.68 AUC. 9% of defendants get conflicting risk labels across 25 equally-accurate models.

German Credit (lending)
- A default-risk model recovers `sex` at 0.73 AUC after it's removed from the features. 10% of applicants get conflicting decisions across equally-accurate models.

## Running it

```bash
pip install -r requirements.txt
python run_export.py
python -m streamlit run app/dashboard.py
```

`run_export.py` runs every audit and writes `results.json`, which the dashboard reads (so it loads without retraining). The "Audit your model" page takes a trained sklearn-compatible classifier (`.pkl` or `.joblib`) and a test CSV and runs the checks live.

## Tests

```bash
PYTHONPATH=. pytest
```

The suite checks that the planted shortcut ranks first, noise features stay near zero, a known proxy is recovered while an independent attribute isn't, small subgroups are skipped, and temporal leakage only fires on overlapping labels.

## Files

```
audit.py             the check functions
model.py             training and scoring
shortcuts.py         plants a known leak for validating shortcut detection
data.py              Adult / Census loader
market_data.py       SPY loader (yfinance)
compas_data.py       COMPAS loader
credit_data.py       German Credit loader
backtest_overfit.py  random-strategy selection-bias demo
scripts/             per-audit demo scripts
run_export.py        runs everything, writes results.json
app/                 Streamlit dashboard and upload page
tests/               pytest suite
```

## Note on COMPAS

This audits a model trained on the public COMPAS dataset, which is what the fairness research literature uses. It is not Northpointe's proprietary scoring system.

## Stack

XGBoost, scikit-learn, Streamlit, Plotly.