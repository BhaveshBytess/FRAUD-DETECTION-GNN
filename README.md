# 🔍 When Graph Neural Networks Fail
## Revisiting Graph Learning on the Elliptic++ Bitcoin Fraud Detection Dataset

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

### 🎯 **TL;DR**

**Graph Neural Networks (GNNs) are supposed to excel at graph-structured data. But on Elliptic++ Bitcoin fraud detection, a simple XGBoost model beats all GNN baselines by 49%.**

This repository investigates **why** — and reveals that pre-computed neighbor aggregates in the features make GNNs redundant.

---

### 🔬 **The Surprising Finding**

> **Main Result:** XGBoost (tabular-only, no graph) achieves **PR-AUC 0.669**, while GraphSAGE (state-of-the-art GNN) achieves only **0.448**.
> 
> **Why?** Features `AF94–AF182` already encode neighbor-aggregated information. Removing them:
> - ✅ GraphSAGE **improves by 24%** (0.448 → 0.556)
> - ✅ XGBoost **drops only 3%** (0.669 → 0.649)
> 
> **Conclusion:** Graph structure *does* add value — but only when features don't already capture it.

---

### 📊 **Performance Comparison**

We trained 7 models using strict temporal splits (no leakage) on the Elliptic++ dataset:

| Model Type | Model | PR-AUC ⭐ | ROC-AUC | F1 | Recall@1% |
|------------|-------|--------:|--------:|----:|----------:|
| 🌳 **Tabular** | **XGBoost** | **0.669** 🥇 | 0.888 | 0.699 | 17.5% |
| 🌳 Tabular | Random Forest | 0.658 🥈 | 0.877 | 0.694 | 17.5% |
| 🕸️ **GNN** | **GraphSAGE** | **0.448** 🥉 | 0.821 | 0.453 | 14.8% |
| 🌐 Tabular | MLP | 0.364 | 0.830 | 0.486 | 9.4% |
| 🕸️ GNN | GCN | 0.198 | 0.763 | 0.249 | 6.1% |
| 🕸️ GNN | GAT | 0.184 | 0.794 | 0.290 | 1.3% |

<div align="center">

![Model Performance Comparison](reports/plots/all_models_comparison.png)

**Figure 1:** XGBoost (tabular) significantly outperforms all GNN baselines on fraud detection.

</div>

> 📌 **Key Insight:** The **49% performance gap** (0.669 vs 0.448) between XGBoost and GraphSAGE led us to investigate feature dominance — see ablation results in [`docs/M7_RESULTS.md`](docs/M7_RESULTS.md).

---

## 🚀 **Quick Start**

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (optional, for GNN training)
- ~2GB disk space for dataset

### Installation & Reproduction

```bash
# 1️⃣ Clone and setup environment
git clone https://github.com/BhaveshBytess/FRAUD-DETECTION-GNN.git
cd FRAUD-DETECTION-GNN
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2️⃣ Download Elliptic++ dataset (NOT included in repo)
# Get from: https://drive.google.com/drive/folders/1MRPXz79Lu_JGLlJ21MDfML44dKN9R08l
# Place these files in: data/Elliptic++ Dataset/
#   ├── txs_features.csv
#   ├── txs_classes.csv
#   └── txs_edgelist.csv

# 3️⃣ Verify data loading
python -m src.data.elliptic_loader --root "data/Elliptic++ Dataset" --check

# 4️⃣ Reproduce results
# Train GNN baseline (GPU recommended, ~15 min)
python -m src.train --config configs/graphsage.yaml

# Train tabular baselines (CPU, ~2 min)
python scripts/run_m5_tabular.py --config configs/m5_xgboost.yaml

# 5️⃣ View results
ls reports/  # Metrics JSON/CSV files
ls reports/plots/  # Figures
```

**Expected Output:** Metrics files in `reports/` matching our published results (±2% variance due to randomness).

---

## 📦 **Dataset**

### Elliptic++ Bitcoin Transaction Network

| Property | Value |
|----------|-------|
| **Nodes** | 203,769 Bitcoin transactions |
| **Edges** | 234,355 transaction flows |
| **Features** | 182 per transaction (local + aggregated) |
| **Labels** | Licit (89%) / Illicit (11%) |
| **Timespan** | 49 timesteps (temporal graph) |
| **Task** | Binary fraud classification |

⚠️ **Dataset NOT included** — Download from [Google Drive](https://drive.google.com/drive/folders/1MRPXz79Lu_JGLlJ21MDfML44dKN9R08l) (public access, no sign-in required)

**Required files:**
```
data/Elliptic++ Dataset/
├── txs_features.csv       (203K rows × 182 features)
├── txs_classes.csv        (node labels)
└── txs_edgelist.csv       (graph edges)
```

**Citation for dataset:**
> Weber, M., et al. (2019). "Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics." *KDD Workshop on Anomaly Detection in Finance*.

---

## 📚 **Project Structure & Documentation**

```
FRAUD-DETECTION-GNN/
├── 📄 README.md                    ← You are here (landing page)
├── 📘 docs/
│   ├── README_FULL.md              ← Complete technical documentation
│   ├── PROJECT_SPEC.md             ← Architecture & acceptance criteria
│   ├── M5_RESULTS_SUMMARY.md       ← Tabular baseline results
│   ├── M7_RESULTS.md               ← 🔬 Feature ablation experiments
│   ├── M8_INTERPRETABILITY.md      ← SHAP & GNN saliency analysis
│   └── M9_TEMPORAL.md              ← Temporal robustness study
├── 📊 reports/
│   ├── metrics_summary.csv         ← All model results
│   └── plots/                      ← Figures (PNG)
├── 📓 notebooks/
│   ├── 03_gcn_baseline.ipynb       ← GNN training workflows
│   ├── 05_m5_tabular.ipynb         ← XGBoost/RF experiments
│   ├── 06_m7_ablation.ipynb        ← Feature ablation analysis
│   ├── 07_interpretability.ipynb   ← SHAP & saliency
│   └── 08_temporal_shift.ipynb     ← Temporal generalization
├── 🧠 src/                         ← Modular source code
│   ├── data/elliptic_loader.py     ← Dataset loader with splits
│   ├── models/                     ← GNN & tabular model definitions
│   ├── train.py                    ← Training script
│   └── eval.py                     ← Evaluation pipeline
├── ⚙️ configs/                     ← YAML configs per model
└── 💾 checkpoints/                 ← Trained model weights
```

### 🔗 **Key Documents**

| Document | Description |
|----------|-------------|
| 📘 [**Full Documentation**](docs/README_FULL.md) | Complete technical README (~10 min read) |
| 📄 [**Project Report**](PROJECT_REPORT.md) | Analysis, discussion, and findings |
| 🔬 [**Feature Ablation**](docs/M7_RESULTS.md) | Why AF94–AF182 explain the GNN gap |
| 🧠 [**Interpretability**](docs/M8_INTERPRETABILITY.md) | SHAP (XGBoost) + saliency (GraphSAGE) |
| ⏱️ [**Temporal Study**](docs/M9_TEMPORAL.md) | Generalization across time windows |

---

## 🏆 **Why This Project Matters**

### Research Contributions
1. **Empirical rigor:** Strict temporal splits, no leakage, reproducible baselines
2. **Unexpected finding:** Tabular models outperform GNNs by 49% on graph data
3. **Root cause analysis:** Feature ablation reveals double-encoding of graph structure
4. **Practical insight:** Graph features can make graph models redundant

### Use Cases
- 🎓 **Students/Researchers:** Reproducible baseline for GNN vs ML comparisons
- 💼 **Data Scientists:** When to use (or avoid) GNNs in fraud detection
- 🏦 **Financial ML Teams:** Feature engineering insights for transaction networks
- 📚 **Educators:** Teaching case study on ablation studies & interpretability

---

## 📖 **Citation**

If you use this code or findings, please cite:

```bibtex
@software{elliptic_gnn_baselines_2025,
  title = {When Graph Neural Networks Fail: Revisiting Graph Learning on the Elliptic++ Dataset},
  author = {Bytes, Bhavesh},
  year = {2025},
  url = {https://github.com/BhaveshBytess/FRAUD-DETECTION-GNN},
  license = {MIT}
}
```

**Machine-readable citation:** See [`CITATION.cff`](CITATION.cff)

---

## 📬 **Contact & License**

**Author:** Bhavesh Bytes  
**Email:** 10bhavesh7.11@gmail.com  
**GitHub:** [@BhaveshBytess](https://github.com/BhaveshBytess)  
**License:** [MIT License](LICENSE)

**Project Status:** ✅ Complete (M1–M9) | **Last Updated:** November 2025

---

<div align="center">

**⭐ If you find this project useful, please consider giving it a star!**

[![GitHub stars](https://img.shields.io/github/stars/BhaveshBytess/FRAUD-DETECTION-GNN?style=social)](https://github.com/BhaveshBytess/FRAUD-DETECTION-GNN/stargazers)

</div>
