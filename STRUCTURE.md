# Repository Structure

This document describes the organization of the repository.

## Root Directory

```
FRAUD-DETECTION-GNN/
├── README.md              # Main project documentation (compact)
├── LICENSE                # MIT License
├── CITATION.cff           # Citation metadata (Zenodo DOI: 10.5281/zenodo.17560930)
├── requirements.txt       # Python dependencies
│
├── src/                   # Source code (models, data loaders, training)
├── configs/               # Configuration files (YAML) for each model
├── scripts/               # Executable scripts (training, evaluation, experiments)
├── notebooks/             # Jupyter notebooks (EDA, experiments, analysis)
│
├── data/                  # Dataset directory (user-provided, NOT in repo)
├── checkpoints/           # Trained model checkpoints
├── reports/               # Metrics, results, plots
├── tests/                 # Unit tests
│
├── docs/                  # Full documentation (specs, results, guides)
└── archive/               # Old/deprecated files (not tracked)
```

## Key Directories

### `src/`
Core implementation: models (GCN, GraphSAGE, GAT, tabular), data loaders, training loops, evaluation metrics.

### `configs/`
YAML configuration files for hyperparameters, model architecture, and training settings.

### `scripts/`
Executable scripts for:
- Training models (`train_gcn.py`, `run_m5_tabular.py`)
- Running experiments (M7 ablations, M8 interpretability, M9 temporal)
- Utility scripts in `scripts/utils/` (dataset checks, fraud rate analysis)

### `notebooks/`
Analysis notebooks (01–08) covering EDA, temporal splits, baseline results, interpretability, and temporal robustness.

### `data/`
**User must download Elliptic++ dataset** and place here:
```
data/Elliptic++ Dataset/
├── txs_features.csv
├── txs_classes.csv
└── txs_edgelist.csv
```
Download link: https://drive.google.com/drive/folders/1MRPXz79Lu_JGLlJ21MDfML44dKN9R08l

### `checkpoints/`
Saved model weights (`.pt` files) for best-performing models from each architecture.

### `reports/`
Metrics (JSON, CSV), plots (PNG), and result summaries organized by milestone (M3–M9).

### `docs/`
In-depth documentation:
- `README_FULL.md` — Comprehensive README with all details
- `PROJECT_SPEC.md` — Technical specification and architecture
- `PROJECT_REPORT.md` — Full research report with findings
- `M7_CAUSALITY_EXPERIMENT.md` — Feature dominance ablation design
- `M7_RESULTS.md` — Causality experiment results
- `M8_INTERPRETABILITY.md` — SHAP and saliency analysis
- `M9_TEMPORAL.md` — Temporal robustness study
- `FEATURE_ANALYSIS.md` — Dataset feature correlations
- `DATA_TYPES_EXPLAINED.md` — Feature schema reference
- `M5_RESULTS_SUMMARY.md` — Tabular baseline results
- `archive/` — Internal working docs (not for public view)

### `archive/`
Old files and internal working documents (ignored by git).

## Clean Commit History

This repository maintains a professional structure:
- All testing/verification scripts in `scripts/utils/`
- All documentation in `docs/`
- No clutter in root (only essential files)
- Comprehensive `.gitignore` to prevent future mess

## For Contributors

When adding files:
- **Code** → `src/` or `scripts/`
- **Configs** → `configs/`
- **Analysis** → `notebooks/`
- **Results** → `reports/`
- **Documentation** → `docs/`
- **Never commit** data files, checkpoints >100MB, or personal notes

---

Last updated: 2025-11-09
