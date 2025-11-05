# TASKS — Single Source of Truth

**Project:** elliptic-gnn-baselines  
**Last Updated:** 2025-11-05  
**Status Legend:** `[ ]` pending | `[~]` in progress | `[x]` done | `[?]` blocked

---

## **M1 — Bootstrap Repo** ✅

**Goal:** Create complete folder structure, requirements.txt, configs, README, and base utilities.

### Steps:
- [x] Create folder tree matching PROJECT_SPEC scaffold
- [x] Add `requirements.txt` with all dependencies
- [x] Create config files (default.yaml, gcn.yaml, graphsage.yaml, gat.yaml)
- [x] Write README.md with project overview
- [x] Create `.gitignore` for Python/Jupyter
- [x] Add `src/utils/seed.py` for reproducibility
- [x] Add `src/utils/metrics.py` for evaluation functions
- [x] Add `src/utils/logger.py` for logging utilities
- [x] Initialize package __init__.py files
- [x] Test: `pip install -r requirements.txt` succeeds

### Done Criteria:
- [x] Folder tree matches PROJECT_SPEC Section 8
- [x] `pip install -r requirements.txt` completes without errors
- [x] All config YAML files created
- [x] README renders correctly
- [x] Verification checklist complete

### Artifacts:
- ✅ Folder structure (all directories created)
- ✅ `requirements.txt` (PyTorch 2.8.0+cpu, Python 3.13.1)
- ✅ `configs/*.yaml` (4 config files)
- ✅ `README.md` (comprehensive project overview)
- ✅ `.gitignore` (Python/Jupyter/data/reports)
- ✅ `src/utils/seed.py`, `metrics.py`, `logger.py`, `explain.py`
- ✅ Package structure with __init__.py files

**Status:** COMPLETE (2025-11-05)

---

## **M2 — Data Loader & Temporal Splits** [~]

**Goal:** Implement `src/data/elliptic_loader.py` to load Elliptic++, create temporal splits, and save `splits.json`.

**Status:** NEXT - Ready to begin

### Steps:
- [ ] Implement `elliptic_loader.py`:
  - [ ] Read `txs_features.csv` + `txs_classes.csv` + `txs_edgelist.csv`
  - [ ] Merge node features and labels
  - [ ] Build `tx_id` → contiguous index mapping
  - [ ] Filter edges to valid nodes only
  - [ ] Create temporal splits (train/val/test) based on timestamp
  - [ ] Filter edges per split (both endpoints must be in same split)
  - [ ] Create PyG `Data` objects with masks
  - [ ] Save `data/elliptic/splits.json`
- [ ] Add `--check` CLI flag to print stats
- [ ] Implement `src/data/splits.py` helper functions
- [ ] Write unit tests in `tests/test_loader.py`
- [ ] Test: `python -m src.data.elliptic_loader --check` works

### Done Criteria:
- [x] `python -m src.data.elliptic_loader --check` prints:
  - Node/edge counts
  - Labeled node counts per split
  - Class balance (fraud/legit)
  - Time range per split
- [x] `splits.json` saved with proper structure
- [x] Unit tests pass (no future edges in train/val)
- [x] Verification checklist complete

### Artifacts:
- `src/data/elliptic_loader.py`
- `src/data/splits.py`
- `data/elliptic/splits.json`
- `tests/test_loader.py`

---

## **M3 — GCN Baseline Notebook**

**Goal:** Implement and train GCN model in a fully reproducible notebook.

### Steps:
- [ ] Create `src/models/gcn.py` with GCN class
- [ ] Create `notebooks/03_gcn_baseline.ipynb`:
  - [ ] Load data using elliptic_loader
  - [ ] Set seeds + deterministic flags
  - [ ] Initialize GCN model
  - [ ] Train with early stopping on val PR-AUC
  - [ ] Evaluate on test set
  - [ ] Calculate metrics (PR-AUC, ROC-AUC, F1, Recall@K)
  - [ ] Generate PR/ROC curve plots
  - [ ] Save checkpoint, metrics.json, plots
  - [ ] Append to metrics_summary.csv
- [ ] Run notebook end-to-end
- [ ] Verify all artifacts created

### Done Criteria:
- [x] Notebook runs fully without errors
- [x] Metrics saved to `reports/metrics.json`
- [x] Plots saved to `reports/plots/`
- [x] Row appended to `reports/metrics_summary.csv`
- [x] Checkpoint saved to `checkpoints/gcn_best.pt`
- [x] No TODOs or placeholders in notebook
- [x] Verification checklist complete

### Artifacts:
- `src/models/gcn.py`
- `notebooks/03_gcn_baseline.ipynb`
- `reports/metrics.json`
- `reports/plots/gcn_pr_curve.png`
- `reports/plots/gcn_roc_curve.png`
- `checkpoints/gcn_best.pt`
- Updated `reports/metrics_summary.csv`

---

## **M4 — GraphSAGE & GAT Notebooks**

**Goal:** Implement GraphSAGE and GAT models and compare performance.

### Steps:
- [ ] Create `src/models/graphsage.py`
- [ ] Create `src/models/gat.py`
- [ ] Create `notebooks/04_graphsage_gat.ipynb`:
  - [ ] Train GraphSAGE model
  - [ ] Train GAT model
  - [ ] Compare metrics across all GNN models
  - [ ] Generate comparison plots
  - [ ] Save artifacts for both models
  - [ ] Append metrics to summary CSV
- [ ] Run notebook end-to-end
- [ ] Verify all artifacts created

### Done Criteria:
- [x] Both models train successfully
- [x] Metrics logged for GraphSAGE and GAT
- [x] Comparison table/plot generated
- [x] All artifacts saved correctly
- [x] Verification checklist complete

### Artifacts:
- `src/models/graphsage.py`
- `src/models/gat.py`
- `notebooks/04_graphsage_gat.ipynb`
- `checkpoints/graphsage_best.pt`
- `checkpoints/gat_best.pt`
- Updated `reports/metrics_summary.csv`
- Comparison plots

---

## **M5 — Tabular Baselines**

**Goal:** Implement non-graph baselines (Logistic Regression, Random Forest, XGBoost, MLP).

### Steps:
- [ ] Create `notebooks/00_baselines_tabular.ipynb`:
  - [ ] Load node features only (no graph structure)
  - [ ] Train Logistic Regression with class weights
  - [ ] Train Random Forest
  - [ ] Train XGBoost
  - [ ] Train MLP (2-3 layers)
  - [ ] Evaluate all models with same metrics
  - [ ] Compare against GNN baselines
  - [ ] Append all metrics to summary CSV
- [ ] Run notebook end-to-end
- [ ] Verify metrics logged

### Done Criteria:
- [x] All 4 tabular models trained and evaluated
- [x] Metrics appended to `reports/metrics_summary.csv`
- [x] Comparison with GNN models documented
- [x] Verification checklist complete

### Artifacts:
- `notebooks/00_baselines_tabular.ipynb`
- Updated `reports/metrics_summary.csv`

---

## **M6 — Final Verification & Readability**

**Goal:** Final checks, documentation polish, and repo cleanup.

### Steps:
- [ ] Create `notebooks/01_eda.ipynb` (exploratory data analysis)
- [ ] Create `notebooks/02_visualize_embeddings.ipynb` (optional)
- [ ] Review all notebooks:
  - [ ] Clear all TODOs/placeholders
  - [ ] Add markdown explanations
  - [ ] Verify all paths are relative
  - [ ] Confirm seeds are set
  - [ ] Check outputs are printed in final cells
- [ ] Update README.md with:
  - [ ] Full project description
  - [ ] Setup instructions
  - [ ] Results summary
  - [ ] Citation for Elliptic++ dataset
- [ ] Write tests in `tests/test_models_shapes.py`
- [ ] Final verification:
  - [ ] All notebooks run end-to-end
  - [ ] All metrics in summary CSV
  - [ ] All plots generated
  - [ ] Repository is clean and professional

### Done Criteria:
- [x] All notebooks are polished and readable
- [x] README is comprehensive
- [x] All tests pass
- [x] Repository ready for portfolio/GitHub showcase
- [x] Verification checklist complete

### Artifacts:
- `notebooks/01_eda.ipynb`
- `notebooks/02_visualize_embeddings.ipynb`
- Updated `README.md`
- `tests/test_models_shapes.py`
- Clean, professional repository

---

## **Escalation Notes**

_None yet._

---

## **Blocked Items**

_None yet._

---

**End of TASKS.md**
