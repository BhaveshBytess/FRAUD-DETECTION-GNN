# M8 — Interpretability & Analysis Plan

**Goal:** Explain *why* tabular models (XGBoost) and GNNs (GraphSAGE) make their predictions after the M7 ablations. We focus on the two decisive configurations surfaced in M7:
- XGBoost trained on the **full** feature set (AF1–AF182).
- GraphSAGE trained on the **local_only** feature set (AF1–AF93), i.e., the run that jumped to 0.556 PR-AUC.

## 1. Prerequisites & Artifacts

| Artifact | Path | Status |
| --- | --- | --- |
| XGBoost training notebook/script | `scripts/run_m5_tabular.py` / `notebooks/05_tabular_baselines.ipynb` | ✅ |
| GraphSAGE local-only metrics | `reports/graphsage_local_only_metrics.json` | ✅ |
| GraphSAGE local-only checkpoint | `checkpoints/graphsage_local_only_best.pt` (`scripts/run_m8_graphsage_local_only.py`) | ✅ |
| Feature definitions | `docs/FEATURE_ANALYSIS.md` | ✅ |
| Interpretability notebook | `notebooks/07_interpretability.ipynb` | ✅ |

To keep interpretability deterministic, rerun:
1. `scripts/run_m5_tabular.py` restricted to the `full` config and save the XGBoost booster (`reports/xgboost_full_model.json`).
2. (Optional) Run Kaggle notebook `06_m7_feature_ablation_kaggle.ipynb` with `FEATURE_CONFIG='local_only'` if you prefer GPU training. Locally we now provide `scripts/run_m8_graphsage_local_only.py` for CPU-based training/checkpointing.

## 2. SHAP for XGBoost (Tabular)

Steps:
1. Load the XGBoost model fitted on AF1–AF182 (use the booster dump if available; otherwise rerun training and save).
2. Use `shap.TreeExplainer` on the validation/test sets (or a representative stratified sample) to compute:
   - Global bar plot of mean |SHAP|.
   - Beeswarm plot to highlight key AF94+ vs AF1–AF93 usage.
3. Save plots to `reports/plots/m8_xgboost_shap_summary.png` and write the top features table to `reports/m8_xgb_shap_top_features.csv`.
4. Note whether AF94–AF182 dominate the importance ranking (expected).

**Current status (run via `scripts/run_m8_interpretability.py`):**
- Booster saved to `reports/models/xgb_full.json`
- Global SHAP CSV saved to `reports/m8_xgb_shap_importance.csv`
- Plot at `reports/plots/m8_xgb_shap_summary.png`

Top features (mean |SHAP| in `reports/m8_xgb_shap_importance.csv`):
1. `Local_feature_53` (largest contribution, negative signed impact)
2. `Local_feature_59`
3. `size`
4. `Aggregate_feature_32`
5. `Local_feature_58`
👉 Plot saved to `reports/plots/m8_xgb_shap_summary.png`.

## 3. GraphSAGE Local-only — Gradient Saliency

PyG in this environment does not ship `GNNExplainer`, so we implemented a deterministic gradient × input fallback:

1. Train/checkpoint GraphSAGE local-only via `scripts/run_m8_graphsage_local_only.py` (saves to `checkpoints/graphsage_local_only_best.pt`, `reports/graphsage_local_only_metrics.json`). CPU PR-AUC ≈ 0.47 (lower than Kaggle but sufficient for saliency).
2. Run `scripts/run_m8_graphsage_saliency.py`:
   - Picks the top-5 high-confidence fraud nodes in the test mask.
   - Computes per-node feature importances via `abs(grad × input)`.
   - Lists the highest-probability neighbors.
   - Saves plots to `reports/plots/m8_graphsage_saliency_node*.png` and JSON to `reports/m8_graphsage_saliency.json`.

## 4. Comparative Analysis (Current Observations)

| Aspect | XGBoost (full) | GraphSAGE (local-only, saliency) |
| --- | --- | --- |
| Artifacts | `reports/m8_xgb_shap_importance.csv`, `reports/plots/m8_xgb_shap_summary.png` | `reports/m8_graphsage_saliency.json`, `reports/plots/m8_graphsage_saliency_node*.png` |
| Dominant features | `Local_feature_53`, `Local_feature_59`, `size`, `Aggregate_feature_32`, `Local_feature_58` | Avg. importance: `Local_feature_90`, `Local_feature_3`, `Local_feature_53`, `Local_feature_55`, `Local_feature_69`, `Local_feature_93`, `Local_feature_80`, `Local_feature_63`, `Local_feature_81`, `Local_feature_92` |
| Narrative | SHAP shows the tree model still depends on late-index locals plus select aggregates. | Saliency highlights AF80–AF93 locals and high-probability neighbors, showing the GNN focuses on raw transaction descriptors + graph context once AF94+ are removed. |

Next steps:
1. Use `notebooks/07_interpretability.ipynb` outputs (plots/JSON) in README & `PROJECT_SUMMARY.md`.
2. Tie these observations into M9/M10 recommendations (monitor AF5x locals for tabular drift, AF8x locals + risky neighbors for GNN).
