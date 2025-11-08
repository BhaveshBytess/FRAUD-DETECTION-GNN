# Elliptic GNN Baselines 🔍

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-complete-success)

> **Fraud Detection on Bitcoin Transactions: Comparing GNN vs Tabular ML Models**

A clean, reproducible implementation comparing **Graph Neural Networks** (GCN, GraphSAGE, GAT) against **tabular ML baselines** (XGBoost, Random Forest, MLP) for fraud detection on the **Elliptic++** dataset. This repository demonstrates temporal-split evaluation, honest metrics reporting (PR-AUC primary), and quantifies the marginal value of graph structure in fraud detection.

**🎯 Main Finding:** Features already encode graph aggregations — GNNs redundant unless features are raw.

---

## 📋 Project Overview

**Goal:** Build reproducible baseline models to answer: *"When does graph structure add value in fraud detection?"*

**Key Features:**
- ✅ Strict temporal splits (no data leakage)
- ✅ Multiple baselines: Tabular (LR, RF, XGBoost, MLP) + GNNs (GCN, GraphSAGE, GAT)
- ✅ Honest evaluation with PR-AUC as primary metric
- ✅ Full reproducibility (seeds, deterministic ops, saved splits)
- ✅ Notebook-driven experiments with saved artifacts
- ✅ Comparative analysis of graph vs tabular approaches

---

## 🎯 Key Findings

### **XGBoost (Tabular) Outperforms GNN Models**

| Rank | Model | Type | PR-AUC | ROC-AUC | F1 | Recall@1% |
|------|-------|------|--------|---------|----|-----------| 
| 🥇 1 | **XGBoost** | Tabular | **0.669** | **0.888** | **0.699** | **0.175** |
| 🥈 2 | Random Forest | Tabular | 0.658 | 0.877 | 0.694 | 0.175 |
| 🥉 3 | **GraphSAGE** | GNN | **0.448** | **0.821** | **0.453** | **0.148** |
| 4 | MLP | Tabular | 0.364 | 0.830 | 0.486 | 0.094 |
| 5 | GCN | GNN | 0.198 | 0.763 | 0.249 | 0.061 |
| 6 | GAT | GNN | 0.184 | 0.794 | 0.290 | 0.013 |
| 7 | Logistic Regression | Tabular | 0.164 | 0.824 | 0.256 | 0.005 |

**Performance Gap:** XGBoost achieves **49% better PR-AUC** than best GNN (GraphSAGE).

### **Why Tabular Features Dominate**
1. **Strong Node Features:** 182 transaction features are highly predictive
2. **Class Imbalance:** ~10% fraud rate affects GNN message passing
3. **Temporal Shift:** Test fraud rate drops to 5.69%, favoring robust tabular models
4. **Efficiency:** XGBoost trains in 2 minutes on CPU vs 15 minutes on GPU for GNNs

### **🔬 Feature Dominance Hypothesis — CONFIRMED (M7)**
> **"Features AF94–AF182 already encode neighbor-aggregated information."**

**M7 Ablation Results:**

| Model | Config | PR-AUC | Δ vs Full | Finding |
|-------|--------|--------|-----------|---------|
| **XGBoost** | Full (AF1–182) | 0.669 | — | Baseline |
| XGBoost | Local only (AF1–93) | 0.648 | −0.021 | **Barely affected** |
| **GraphSAGE** | Full (AF1–182) | 0.448 | — | Redundant encoding |
| GraphSAGE | Local only (AF1–93) | **0.556** | **+0.108** | **GNN unlocked!** |

**Key Insight:** Removing AF94–AF182 causes:
- ✅ **GraphSAGE improves 24%** (0.448 → 0.556) — now learns from graph structure
- ✅ **XGBoost barely drops 3%** (0.669 → 0.648) — local features sufficient
- ✅ **Hypothesis CONFIRMED:** Features pre-encode graph signals

**Correlation Evidence:**
- Neighbor averages correlate **0.74–0.89** with aggregate features
- Manual graph metrics correlate **0.63–0.65** with aggregates
- AF94–AF182 are literally pre-computed neighbor aggregations

**M8 Interpretability Results:**
- **SHAP Analysis:** XGBoost heavily relies on AF94+ aggregate features
- **GNN Saliency:** GraphSAGE (local-only) focuses on raw transaction patterns
- Models learn *different* representations: features vs graph structure

**M9 Temporal Robustness:**
- XGBoost maintains 0.67–0.78 PR-AUC across time shifts
- GraphSAGE (local) improves 0.41 → 0.56 with earlier training windows
- **Finding:** GNNs benefit from structural drift when features don't pre-encode graph

**Recommendation:** 
- **Production:** Use XGBoost (fast, interpretable, CPU-friendly)
- **Research:** Use GNNs when features *don't* already encode graph aggregations
- **Insight:** Graph structure **is** valuable — dataset just pre-captured it

---

## 🗂️ Repository Structure

```
elliptic-gnn-baselines/
│
├── data/
│   └── elliptic/              # Elliptic++ dataset (user-provided)
│
├── notebooks/
│   ├── 00_baselines_tabular.ipynb
│   ├── 01_eda.ipynb
│   ├── 02_visualize_embeddings.ipynb
│   ├── 03_gcn_baseline.ipynb
│   └── 04_graphsage_gat.ipynb
│
├── src/
│   ├── data/                  # Data loaders & split utilities
│   ├── models/                # GNN model implementations
│   └── utils/                 # Metrics, logging, seed utilities
│
├── configs/                   # YAML configuration files
├── reports/                   # Metrics, plots, summaries
├── checkpoints/               # Model checkpoints
├── tests/                     # Unit tests
└── docs/                      # Project documentation

```

---

## 🚀 Setup

### Prerequisites
- Python >= 3.10
- CUDA-capable GPU (recommended) or CPU

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/BhaveshBytess/FRAUD-DETECTION-GNN.git
cd FRAUD-DETECTION-GNN
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Prepare the dataset:**
   - Obtain the Elliptic++ dataset
   - Place files in `data/elliptic/`:
     - `txs_features.csv`
     - `txs_classes.csv`
     - `txs_edgelist.csv`

---

## 📊 Usage

### 1. Data Preparation & Splits
```bash
python -m src.data.elliptic_loader --check
```
This validates the dataset and creates temporal splits saved to `data/elliptic/splits.json`.

### 2. Run Experiments

**Tabular Baselines:**
```bash
jupyter notebook notebooks/00_baselines_tabular.ipynb
```

**GNN Baselines:**
```bash
jupyter notebook notebooks/03_gcn_baseline.ipynb
jupyter notebook notebooks/04_graphsage_gat.ipynb
```

### 3. View Results
All metrics are logged to `reports/metrics_summary.csv` with plots saved in `reports/plots/`.

---

## 📈 Metrics

We evaluate models using:
- **PR-AUC** (Primary metric for imbalanced data - ~10% fraud rate)
- **ROC-AUC**
- **F1 Score** (threshold selected on validation set)
- **Recall@K** (K ∈ {0.5%, 1%, 2%} of test size)

### Dataset Statistics
- **Total Nodes:** 203,769 transactions
- **Labeled Nodes:** 46,564 (22.9%)
- **Features:** 182 per transaction
- **Fraud Rate:** Train 10.88%, Val 11.53%, Test 5.69%
- **Temporal Splits:** Train (≤29), Val (≤39), Test (>39)

---

## 🧪 Models Implemented

### Tabular Baselines (No Graph Structure)
- Logistic Regression
- Random Forest
- XGBoost
- MLP (2-3 layers)

### GNN Baselines
- **GCN** (Graph Convolutional Network)
- **GraphSAGE** (Graph Sample and Aggregate)
- **GAT** (Graph Attention Network)

---

## 🔬 Reproducibility

All experiments are fully reproducible:
- Fixed random seeds (42)
- Deterministic PyTorch operations
- Saved data splits (`splits.json`)
- Version-controlled configurations

To reproduce results:
```python
from src.utils.seed import set_all_seeds
set_all_seeds(42)
```

---

## 📚 Dataset Citation

This project uses the **Elliptic++** dataset. Please cite the original paper:

```
Weber, M., Domeniconi, G., Chen, J., Weidele, D. K. I., Bellei, C., Robinson, T., & Leiserson, C. E. (2019).
Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics.
arXiv preprint arXiv:1908.02591.
```

---

## 📂 Key Files

### Best Model
- `reports/xgboost_metrics.json` - Best overall performance (0.669 PR-AUC)
- `reports/graphsage_metrics.json` - Best GNN performance (0.448 PR-AUC)

### Documentation
- `docs/M5_RESULTS_SUMMARY.md` - Detailed comparative analysis
- `PROJECT_SUMMARY.md` - Complete project overview
- `TASKS.md` - Development tracker

### Notebooks
- `notebooks/05_tabular_baselines.ipynb` - XGBoost, RF, LR, MLP training
- `notebooks/04_graphsage_gat_kaggle.ipynb` - GNN model training
- `notebooks/03_gcn_baseline.ipynb` - GCN baseline

---

## 🎯 Project Status

- [x] **M1:** Repository scaffold ✅
- [x] **M2:** Data loader & temporal splits ✅
- [x] **M3:** GCN baseline (PR-AUC: 0.198) ✅
- [x] **M4:** GraphSAGE (PR-AUC: 0.448) & GAT (PR-AUC: 0.184) ✅
- [x] **M5:** Tabular baselines (XGBoost: 0.669 PR-AUC) ✅
- [x] **M6:** Documentation & comparative analysis ✅
- [x] **M7:** Causality & Feature Dominance — **HYPOTHESIS CONFIRMED** ✅
- [x] **M8:** Interpretability (SHAP + GNN saliency) ✅
- [x] **M9:** Temporal Robustness Study ✅
- [ ] **M10:** Final Portfolio Polish ⏳

### Current Focus: M10 — Final Wrap
**Completed Analysis:**
- ✅ M7: Feature dominance confirmed (AF94–AF182 double-encode graph)
- ✅ M8: SHAP shows XGBoost uses aggregates; GNN saliency on local features
- ✅ M9: Temporal robustness measured (XGBoost stable, GNN improves with drift)

**Remaining Tasks:**
- Documentation polish (README ✅, PROJECT_REPORT pending)
- Repository cleanup
- Final release preparation

---

## 🔬 Technical Highlights

**This project demonstrates:**
- ✅ Complete GNN pipeline implementation (PyTorch Geometric)
- ✅ Fair comparison methodology (same splits, metrics, seeds)
- ✅ Temporal validation (no future data leakage)
- ✅ Class imbalance handling (weighted loss, PR-AUC metric)
- ✅ Scientific rigor (reproducible, documented, validated)
- ✅ Business judgment (cost-benefit analysis, production recommendations)

---

## 📝 License

This project is for educational and demonstration purposes.

---

## 🤝 Contributing

This is a portfolio/demonstration project. Feel free to fork and adapt for your own use cases.

---

## 📧 Contact

**Repository:** [FRAUD-DETECTION-GNN](https://github.com/BhaveshBytess/FRAUD-DETECTION-GNN)  
**GitHub:** [@BhaveshBytess](https://github.com/BhaveshBytess)

---

## 📖 Further Reading

**Complete Documentation:**
- 📊 **[PROJECT_REPORT.md](PROJECT_REPORT.md)** - Publication-style full report (recommended)
- 📝 **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - Comprehensive project overview
- 🔬 **[docs/M7_RESULTS.md](docs/M7_RESULTS.md)** - Feature dominance ablation results
- 🧠 **[docs/M8_INTERPRETABILITY.md](docs/M8_INTERPRETABILITY.md)** - SHAP + GNN saliency analysis
- ⏱️ **[docs/M9_TEMPORAL.md](docs/M9_TEMPORAL.md)** - Temporal robustness findings
- 📐 **[docs/PROJECT_SPEC.md](docs/PROJECT_SPEC.md)** - Technical specifications
- 📋 **[TASKS.md](TASKS.md)** - Development tracker

**Key Artifacts:**
- `reports/m7_*.csv` - Ablation experiment results
- `reports/m8_*.csv/json` - Interpretability artifacts
- `reports/m9_temporal_results.csv` - Temporal robustness data
- `notebooks/03-08_*.ipynb` - Reproducible experiments

---

## 📝 License

MIT License - See LICENSE file for details.

This project is for educational and demonstration purposes.

---

## 🎓 Citation

If you use this work, please cite:

```bibtex
@misc{elliptic-gnn-2025,
  title={Elliptic++ Fraud Detection: When Do Graph Neural Networks Add Value?},
  author={[Your Name]},
  year={2025},
  url={https://github.com/BhaveshBytess/FRAUD-DETECTION-GNN}
}
```

---

## 🤝 Contributing

This is a portfolio/demonstration project. Feel free to fork and adapt for your own use cases.

Contributions welcome for:
- Additional GNN architectures (TGN, TGAT, etc.)
- Other fraud detection datasets
- Improved visualizations
- Documentation improvements

---

## 📧 Contact

**Repository:** [FRAUD-DETECTION-GNN](https://github.com/BhaveshBytess/FRAUD-DETECTION-GNN)  
**GitHub:** [@BhaveshBytess](https://github.com/BhaveshBytess)

---

**⭐ If you find this project useful, please star the repository!**

**Project Status:** ✅ Complete (M1-M9) | Last Updated: 2025-11-08
