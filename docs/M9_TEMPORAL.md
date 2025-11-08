# M9 — Temporal Robustness Study

**Artifacts**
- Results CSV: `reports/m9_temporal_results.csv`
- Script: `scripts/run_m9_temporal_shift.py`
- Kaggle notebook: `notebooks/08_temporal_shift.ipynb`

## Scenario Summary

| Scenario | Model | PR-AUC | ROC-AUC | F1 | Recall@1% | Recall@0.5% | Recall@2% |
|---------|-------|--------|---------|----|-----------|-------------|-----------|
| baseline | XGBoost | 0.669 | 0.890 | 0.694 | 0.175 | — | — |
| baseline | GraphSAGE (local) | 0.413 | 0.797 | 0.382 | 0.162 | 0.083 | 0.275 |
| shift_mid (train≤24, val≤34, test>34) | XGBoost | 0.785 | 0.930 | 0.753 | 0.153 | — | — |
| shift_mid | GraphSAGE (local) | 0.534 | 0.853 | 0.546 | 0.146 | 0.072 | 0.272 |
| shift_long (train≤20, val≤30, test>30) | XGBoost | 0.731 | 0.856 | 0.599 | 0.123 | — | — |
| shift_long | GraphSAGE (local) | 0.557 | 0.843 | 0.603 | 0.115 | 0.057 | 0.224 |

## Observations

- **XGBoost** retains strong PR-AUC even when the train window moves earlier (0.67 → 0.78 → 0.73). Mild oscillation indicates the engineered aggregates generalize across time.
- **GraphSAGE (local-only)** improves as the temporal gap widens (0.41 → 0.53 → 0.56). Earlier training windows benefit the GNN because the later test sets exhibit stronger structural drift that the graph model can capture once trained purely on raw features.
- Recall@0.5%/1% remain competitive for GraphSAGE, while XGBoost does not log those metrics in the current script—future work could add custom recall calculations for parity.

## Next Steps / Recommendations

1. **Metric parity:** extend the XGBoost logging to compute Recall@k so both families can be compared at low operating points.
2. **Visualization:** add a line chart (PR-AUC vs. scenario) for the README to show relative robustness.
3. **Automation:** wire `notebooks/08_temporal_shift.ipynb` into the README so others can rerun the experiment on Kaggle easily.
