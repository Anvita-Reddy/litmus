# Litmus

![tests](https://github.com/Anvita-Reddy/litmus/actions/workflows/test.yml/badge.svg)

A model-auditing toolkit that checks whether a classifier's accuracy is real, or
inflated, unfair, or arbitrary. Litmus runs five checks against a trained model
and reports where the headline number breaks down. Each check is validated on a
controlled benchmark (Adult / Census Income) with planted ground truth, then
pointed at financial data to show it generalizes.

## Checks

- `feature_reliance` - permutation importance; ranks what the model actually
  depends on and auto-discovers shortcuts without being told where to look.
- `proxy_detection` - trains a probe to reconstruct a protected attribute from
  the remaining features; measures whether dropping it actually removed it.
- `worst_group` - per-subgroup accuracy; surfaces failures the average hides.
- `predictive_multiplicity` - trains many equally-accurate models and measures
  how often they disagree on the same individual.
- `temporal_leakage` - random split vs walk-forward; catches inflated backtest
  scores on time-ordered data.

## Findings

### Adult / Census Income

- A model hits 96.9% accuracy by leaning on one planted leak. Litmus ranks that
  feature first by reliance (0.20 vs 0.03 for the next). Permuting it drops
  accuracy to 76.7%, below the 87.3% honest baseline: the model now trusts noise.
- Dropping `sex` from the features is cosmetic. A probe reconstructs it at 0.94
  AUC from `relationship`, `occupation`, and `marital-status`.
- 25 equally-accurate models (87.0-87.4%) give conflicting decisions to 2.5% of
  individuals, and 1.9x more often for men than women.

### Market data (SPY, next-day direction)

- Clean causal features: no leak (random vs walk-forward gap = 0.00).
- Overlapping 20-day labels: the random-split backtest reads 65.9% while honest
  walk-forward reads 57.8%. The 8-point edge is entirely leakage.
- On this low-signal task 25 equally-accurate models disagree on 40.8% of days,
  and none beat a constant "always up" baseline.

## Run

```
pip install -r requirements.txt

python run_stress_test.py      # shortcut auto-discovery, then collapse
python run_audit_baseline.py   # proxy + worst-group on the honest model
python run_multiplicity.py     # predictive multiplicity
python run_market_audit.py     # temporal leakage + multiplicity on market data
```

## Layout

```
data.py          load + encode the Adult dataset
model.py         train / score, RANDOM_STATE
shortcuts.py     planted shortcuts for validating the checks
audit.py         the five checks
market_data.py   price data -> next-day-direction task
run_*.py         entry points
```
