# FEATURE_ANALYSIS — Elliptic++ Columns

M7 requires a precise map of which Elliptic++ features are purely local vs.
neighbor-aggregated. The current `txs_features.csv` exposes **184** columns that
fall into three disjoint families:

| Group                     | Prefix / Columns           | Count | Notes |
|--------------------------|----------------------------|------:|-------|
| Local transaction stats  | `Local_feature_1`–`Local_feature_93` | 93 | Raw descriptors tied to the focal transaction only. |
| Aggregated neighbor stats| `Aggregate_feature_1`–`Aggregate_feature_72` | 72 | Suspected neighbor-aggregation outputs (matches AF94–AF182 conceptually, only 72 columns in this release). |
| Structural/manual stats  | Degree/BTC/address columns (`in_txs_degree`, `total_BTC`, …, `out_BTC_total`) | 17 | Hand-crafted graph features already provided in the CSV (excludes ID/time columns). |

The helper module `src/data/feature_groups.py` now codifies these lists so that
scripts/notebooks can request repeatable subsets (e.g., `local_only`,
`aggregate_only`, `local_plus_structural`).

## Redundancy Signals (Aggregated vs Structural)

To test whether the aggregated columns already encode graph structure, we
computed Pearson correlations between every structural column and every aggregate
column across all **203,769** labeled transactions:

| Structural metric   | Aggregate feature      | Pearson r |
|---------------------|------------------------|-----------|
| `total_BTC`         | `Aggregate_feature_1`  | **0.638** |
| `total_BTC`         | `Aggregate_feature_4`  | 0.629 |
| `total_BTC`         | `Aggregate_feature_2`  | 0.528 |
| `total_BTC`         | `Aggregate_feature_38` | 0.491 |
| `total_BTC`         | `Aggregate_feature_39` | 0.472 |
| `total_BTC`         | `Aggregate_feature_40` | 0.467 |
| `num_input_addresses` | `Aggregate_feature_26` | 0.320 |
| `num_output_addresses`| `Aggregate_feature_68` | 0.301 |
| `num_input_addresses` | `Aggregate_feature_21` | 0.210 |
| `in_txs_degree`       | `Aggregate_feature_26` | 0.191 |

Meanwhile, the correlation between the **mean local feature vector** and the
**mean aggregate feature vector** is only **0.068**, reinforcing that the
aggregates align far more with structural/topological statistics than with the
local descriptors. This quantitative check supports the feature-dominance
hypothesis articulated in `docs/M7_CAUSALITY_EXPERIMENT.md`.

## Next Steps

* Use `CONFIGS` in `scripts/run_m7_tabular_ablation.py` to train the tabular
  baselines across `full`, `local_only`, `aggregate_only`, and
  `local_plus_structural` subsets.
* Mirror the same feature subsets for the Kaggle-hosted GNN experiments so the
  causality comparison stays apples-to-apples.
