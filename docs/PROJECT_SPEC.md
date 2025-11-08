# PROJECT_SPEC

## 0) Purpose (single source of truth)

Define the **what** of this project: scope, dataset, metrics, artifacts, repo scaffold, and acceptance criteria. This document is the canonical reference for requirements. **All tasks must align with this spec.**

---

## 1) Goal & Scope

**Project:** `elliptic-gnn-baselines`
**Goal:** Implement clean, reproducible **GNN baselines** (GCN, GraphSAGE, GAT) on **Elliptic++**, with strict temporal splits and honest evaluation.
**Audience:** Recruiters, collaborators, future-you.
**Deliverable type:** Portfolio/demo repo — readable notebooks first, reusable `src/` utilities second.

**In scope**

* Node classification (fraud / non-fraud) on the Elliptic++ graph.
* Baselines: Logistic Regression, Random Forest / XGBoost, MLP (tabular, no graph).
* GNNs: GCN, GraphSAGE, GAT (static snapshot with strict temporal splitting).
* Metrics: PR-AUC (primary), ROC-AUC, F1 (val-selected threshold), Recall@K.
* Notebook-driven experiments + saved artifacts (plots, metrics, checkpoints).
* Reproducibility: fixed seeds, deterministic ops, saved splits.

**Out of scope (for this repo)**

* Temporal memory models (TGN/TGAT), SpotTarget, TRDGNN, CUSP/curvature, hypergraphs, hetero-GNNs, advanced explainers.
* Any synthetic/mock data.
* Productionization (APIs/serving).

---

## 2) Dataset (Elliptic++)

**Identity:** Elliptic++ Bitcoin transaction graph (nodes = transactions; edges = directed flows).
**Location:** `data/Elliptic++ Dataset/` (local only; user provides files).
**Download:** https://drive.google.com/drive/folders/1MRPXz79Lu_JGLlJ21MDfML44dKN9R08l
**Required files**

* `txs_features.csv` — columns:

  * `txid` (int/str unique),
  * Timestep column (int),
  * `Local_feature_1` to `Local_feature_93`, `Aggregate_feature_1` to `Aggregate_feature_89`, and additional features (total 182 features).
* `txs_classes.csv` — columns:

  * `txid`,
  * `class` (int: 1=fraud/illicit, 2=licit/legit, 3=unknown/unlabeled).
* `txs_edgelist.csv` — columns:

  * `txId1`, `txId2` (directed edges).

**Data policy**

* **No synthetic data.** If files are missing or columns mismatch, stop and request the correct path.
* All notebooks must verify file presence before running.
* Any preprocessing must be deterministic and logged.

---

## 3) Temporal Split (no leakage)

We mimic deployment by ensuring **no future information** contaminates training.

1. Sort nodes by `timestamp`.
2. Choose cutoffs (from config):

   * `t_train_end < t_val_end < t_test_end_max`.
3. Split membership:

   * **Train:** `timestamp ≤ t_train_end`
   * **Val:** `t_train_end < timestamp ≤ t_val_end`
   * **Test:** `timestamp > t_val_end`
4. For each split’s graph, include **only edges whose both endpoints** fall in that split.
5. Save `data/Elliptic++ Dataset/splits.json` with counts and boundaries.

---

## 4) Preprocessing & Features

* Convert timestamps to a single numeric axis if needed (e.g., month index).
* Map `tx_id` → contiguous node indices `[0..N-1]`.
* Filter edges to known nodes; coalesce duplicates; (optional) store weights.
* Standardize/normalize features if beneficial; record scalers (fit on **train only**).
* Maintain masks: `train_mask`, `val_mask`, `test_mask` for labeled nodes only.

---

## 5) Models

### 5.1 Tabular baselines (no graph)

* Logistic Regression (with class weights).
* Random Forest / XGBoost.
* MLP (2–3 layers).

**Input:** node feature vectors for nodes in each split (no neighbor info).
**Output:** probabilities for class 1 (fraud).

### 5.2 GNN baselines (PyTorch Geometric)

* **GCN** (2–3 conv layers)
* **GraphSAGE** (2–3 conv layers)
* **GAT** (2–3 conv layers; configurable heads)

**Common:**

* `in_channels`, `hidden_channels`, `out_channels=2`, `num_layers`, `dropout`.
* Activation: ReLU; final layer returns logits `[N, 2]`.

---

## 6) Training & Evaluation

**Loss**

* Either `CrossEntropyLoss` on logits `[N, 2]` **or** `BCEWithLogitsLoss` on positive logit — pick one and be consistent.
* Use `pos_weight` or class weights computed from **train** labels.

**Optimization**

* Adam, `lr` (default 1e-3), `weight_decay` (default 5e-4).
* Early stopping on **val PR-AUC** with patience (default 15 epochs).

**Evaluation protocol**

* Select threshold on **val** to maximize F1; reuse on **test**.
* Report on **test**:

  * PR-AUC (primary),
  * ROC-AUC,
  * F1 at val-selected threshold,
  * Recall@K where K ∈ {0.5%, 1%, 2%} of test size (rank by P(fraud)).

**Artifacts**

* `checkpoints/model_best.pt`
* `reports/metrics.json` (per split)
* `reports/plots/*.png` (PR & ROC curves; optional embedding plots)
* Append a row to `reports/metrics_summary.csv`:

  * `timestamp, experiment, model, split, pr_auc, roc_auc, f1, recall@1%`

---

## 7) Reproducibility

Always:

```python
from src.utils.seed import set_all_seeds
set_all_seeds(seed)

import torch
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
```

* Save `splits.json`, scaler params (if any), library versions (PyTorch, PyG).
* Avoid absolute paths; use project-relative paths.

---

## 8) Repository Scaffold

```
elliptic-gnn-baselines/
│
├── data/
│   └── elliptic/                  # user-provided dataset (nodes.csv, edges.csv)
│
├── notebooks/
│   ├── 00_baselines_tabular.ipynb
│   ├── 01_eda.ipynb
│   ├── 02_visualize_embeddings.ipynb
│   ├── 03_gcn_baseline.ipynb
│   └── 04_graphsage_gat.ipynb
│
├── src/
│   ├── data/
│   │   ├── elliptic_loader.py     # builds Data objects + masks + splits.json
│   │   └── splits.py              # split helpers, validators
│   ├── models/
│   │   ├── gcn.py
│   │   ├── graphsage.py
│   │   └── gat.py
│   ├── utils/
│   │   ├── metrics.py
│   │   ├── seed.py
│   │   ├── logger.py
│   │   └── explain.py             # stub, not used yet
│   ├── train.py                   # CLI training (uses config)
│   └── eval.py                    # CLI eval + metrics_summary append
│
├── configs/
│   ├── default.yaml
│   ├── gcn.yaml
│   ├── graphsage.yaml
│   └── gat.yaml
│
├── tests/
│   ├── test_loader.py             # shapes, masks, no-future-edges
│   └── test_models_shapes.py      # forward returns [N,2]
│
├── reports/
│   ├── plots/
│   └── metrics_summary.csv
│
├── docs/
│   ├── PROJECT_SPEC.md
│   └── AGENT.MD                   # operational rules (separate file you wrote)
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 9) Configuration (YAML)

**Base keys**

```yaml
experiment: "elliptic-gnn-baselines"
seed: 42
device: "cuda"   # or "cpu"

data:
  root: "data/elliptic"
  nodes: "nodes.csv"
  edges: "edges.csv"
  cache_processed: true

splits:
  # Either integer timesteps or ISO8601; loader must handle both
  train_end: "2017-06-01"
  val_end:   "2017-07-01"

model:
  name: "gcn"     # ["gcn","graphsage","gat"]
  in_channels: 166
  hidden_channels: 128
  num_layers: 2
  dropout: 0.4
  gat_heads: 4

train:
  epochs: 100
  batch_size: null       # null = full batch
  lr: 0.001
  weight_decay: 0.0005
  early_stopping_patience: 15

eval:
  recall_k_fracs: [0.005, 0.01, 0.02]
  save_plots: true

logging:
  out_dir: "reports"
```

Each model-specific config overrides `model.name`, `hidden_channels`, `num_layers`, `dropout`, and `gat_heads` where relevant.

---

## 10) Metrics & File Formats

**`reports/metrics.json` (per split)**

```json
{
  "pr_auc": 0.8421,
  "roc_auc": 0.9102,
  "best_f1": 0.6123,
  "threshold": 0.438,
  "recall@1%": 0.451
}
```

**Append row to `reports/metrics_summary.csv`:**

```
timestamp,experiment,model,split,pr_auc,roc_auc,f1,recall@1%
1730843200,elliptic-gnn-baselines,GCN,test,0.842100,0.910200,0.612300,0.451000
```

---

## 11) Acceptance Criteria (per milestone)

**M1 — Bootstrap**

* Repo scaffold matches Section 8.
* `pip install -r requirements.txt` succeeds.

**M2 — Data loader & splits**

* `python -m src.data.elliptic_loader --check` prints node/edge counts, labeled nodes, class balance, time range.
* `splits.json` saved; unit tests confirm **no future edges** in train/val/test.

**M3 — GCN notebook**

* `03_gcn_baseline.ipynb` runs top-to-bottom, saves checkpoint, metrics, plots, and appends `metrics_summary.csv`.

**M4 — GraphSAGE & GAT notebook**

* `04_graphsage_gat.ipynb` runs both models and appends metrics.

**M5 — Tabular baselines**

* `00_baselines_tabular.ipynb` logs LR/XGB/MLP metrics to `metrics_summary.csv`.

**M6 — Readability**

* Notebooks use markdown explanations; no TODOs/placeholders; paths are relative; seeds set.

---

## 12) Risks & Pitfalls (and how we avoid them)

* **Data leakage via edges** → We filter edges per split so **both endpoints** must belong to the same split.
* **Imbalanced labels** → Use class weights/`pos_weight`; PR-AUC is primary.
* **Unlabeled nodes** → Compute loss/metrics **only** on labeled masks.
* **Non-reproducible results** → Seeds + deterministic ops, saved splits, version logs.
* **Overfitting** → Early stopping on val PR-AUC; keep models small and simple.
* **Broken notebooks** → Verification checklists + artifact paths printed in final cells.

---

## 13) Roadmap (future repos, not here)

* Temporal GNNs (TGN/TGAT) with event time encodings.
* Heterogeneous / Hypergraph modeling.
* Training disciplines (SpotTarget / TRDGNN sampling).
* Explainability (GNNExplainer/PGExplainer), auditor reports.
* Cost/latency benchmarks for deployable candidates.

---

## 14) License & Acknowledgements

* Respect Elliptic++ dataset licensing/terms.
* Cite the dataset in `README.md` and any write-ups.
* This repo is educational/demonstrative.

---

**End of `PROJECT_SPEC.md`.**
