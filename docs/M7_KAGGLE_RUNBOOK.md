# M7 Kaggle Runbook — GNN Feature Ablations

Goal: run the four feature configurations that we already tested on the tabular models (**full, local-only, aggregate-only, local+structural**) using the GNN baselines (GraphSAGE primary, GCN optional) on Kaggle GPUs, then pull the metrics back into `reports/`.

## 1. Pre-flight Checklist

1. **Freeze repo snapshot**  
   - Zip the repository (exclude `.git`, `.venv`, `reports/`, `checkpoints/`).  
   - Upload to Kaggle under *Datasets → New Dataset → “elliptic-gnn-code”*.  
   - Folder layout inside Kaggle dataset should preserve `src/`, `configs/`, `notebooks/`, etc.

2. **Upload Elliptic++ data**  
   - Zip `data/Elliptic++ Dataset/` as `elliptic-data.zip` (contains `txs_features.csv`, `txs_classes.csv`, `txs_edgelist.csv`, `splits.json`).  
   - Create a second Kaggle dataset named e.g. `elliptic-fraud-data`. Kaggle will expose it under `/kaggle/input/elliptic-fraud-data/`.

3. **Notebook hardware**  
   - Open https://www.kaggle.com/code → *New Notebook*.  
   - In Settings: Accelerator = **GPU (T4)**, Internet = Off, Output = On. GPU runtime is required for GraphSAGE/GAT.

## 2. Notebook Template

Use `notebooks/04_graphsage_gat_kaggle.ipynb` as the base. In the Kaggle editor:

1. Copy the entire notebook contents from the repo into the Kaggle notebook.  
2. Add the repo dataset and the data dataset via the right-hand “Add data” panel:
   - `elliptic-gnn-code`
   - `elliptic-fraud-data`
3. At the top of the notebook (after `sys.path` setup), insert the config cell below to switch feature subsets:

```python
# --- M7 feature config ---
from src.data.feature_groups import resolve_group

FEATURE_CONFIG = "full"  # options: full, local_only, aggregate_only, local_plus_structural

FEATURE_MAP = {
    "full": None,
    "local_only": resolve_group("local"),
    "aggregate_only": resolve_group("aggregate"),
    "local_plus_structural": resolve_group("local_plus_structural"),
}

feature_subset = FEATURE_MAP[FEATURE_CONFIG]
dataset = EllipticDataset(
    root="/kaggle/input/elliptic-fraud-data/Elliptic++ Dataset",
    feature_subset=feature_subset,
)
data = dataset.load(verbose=True)
print(f"[M7] Feature config: {FEATURE_CONFIG} | dims={data.x.shape[1]}")
```

4. Leave the rest of the notebook unchanged. The PyG models will now train on the requested subset.

## 3. Run Matrix (per configuration)

| Config Key             | Feature Set                              | Models to Run | Notes |
|------------------------|------------------------------------------|---------------|-------|
| `full`                 | AF1–AF182                                | GraphSAGE + (optional GCN) | Baseline sanity check |
| `local_only`           | AF1–AF93                                 | GraphSAGE + (optional GCN) | Removes all suspected neighbor aggregates |
| `aggregate_only`       | AF94–AF182                               | GraphSAGE (GCN often diverges) | Expect underfitting; shorter patience (30 epochs) |
| `local_plus_structural`| AF1–AF93 + structural stats (17 cols)    | GraphSAGE     | Tests hand-crafted graph stats without AF94+ |

Recommended order: run GraphSAGE for all four configs first (primary comparison), then re-run `full` vs `local_only` with GCN if GPU time allows.

## 4. Hyperparameter Notes

- GraphSAGE settings in the Kaggle notebook already match the PR-AUC 0.448 baseline (128 hidden units, 2 layers, dropout 0.4, patience 20). Keep them fixed for comparability.  
- For `aggregate_only`, set `EARLY_STOPPING_PATIENCE = 10` to avoid wasting GPU minutes—these features are weaker.  
- Learning rate remains 1e-3, weight decay 5e-4.

## 5. Artifact Expectations

Each run writes into `/kaggle/working/` following the repo conventions:

```
reports/
  graphsage_full_metrics.json
  graphsage_local_only_metrics.json
  graphsage_aggregate_only_metrics.json
  graphsage_local_plus_structural_metrics.json
  metrics_summary.csv   # appended per run
  plots/
    graphsage_full_pr_roc.png
    graphsage_full_training.png
checkpoints/
  graphsage_full_best.pt
  ...
```

After a notebook finishes:

1. Download the `reports/` and `checkpoints/` subfolders you need (at minimum, metrics JSON + training plots + checkpoints).  
2. Back on your workstation, place them under the same relative paths (`reports/`, `checkpoints/`).  
3. Append the results to `reports/m7_tabular_ablation.csv` or create `reports/m7_gnn_ablation.csv` (one row per config per model) so TASKS.md can reference a single artifact.

## 6. Logging in TASKS.md

For each config run:

```
- [x] GraphSAGE — CONFIG_NAME (Kaggle) → PR-AUC=X.XXX (reports/graphs...json)
```

Include the Kaggle notebook URL + runtime so we can trace the execution environment.

## 7. Failure Recovery

- **OOM / 12GB cap**: reduce `batch_size` (if using neighbor loaders) or decrease hidden size to 64.  
- **Patience never triggers**: shorten epochs to 60 and patience to 10; we only need relative comparisons.  
- **Unexpectedly high PR-AUC (>0.9)**: stop and open a LeakageSuspect note per AGENT.MD.

## 8. Deliverables Recap

1. Kaggle notebook links (one per config or one notebook that re-runs all four with a loop).  
2. Metrics + plots + checkpoints copied back under `reports/` & `checkpoints/`.  
3. Updated `TASKS.md` + `docs/M7_CAUSALITY_EXPERIMENT.md` with the observed deltas.  
4. Short findings summary (do GNNs gain ground once AF94–AF182 are removed?).

This runbook should be all you need to execute M7 remotely without further local changes. Ping before running if you want me to script a Kaggle loop notebook to speed up the process.
