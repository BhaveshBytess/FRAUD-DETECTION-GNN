# Elliptic GNN Baselines 🔍

> **Fraud Detection on Bitcoin Transactions using Graph Neural Networks**

A clean, reproducible implementation of GNN baselines (GCN, GraphSAGE, GAT) for fraud detection on the **Elliptic++** dataset. This repository demonstrates temporal-split evaluation, honest metrics reporting (PR-AUC primary), and best practices for graph-based anomaly detection.

---

## 📋 Project Overview

**Goal:** Build reproducible baseline models for transaction fraud detection using Graph Neural Networks on the Elliptic++ Bitcoin transaction graph.

**Key Features:**
- ✅ Strict temporal splits (no data leakage)
- ✅ Multiple baselines: Tabular (LR, RF, XGBoost, MLP) + GNNs (GCN, GraphSAGE, GAT)
- ✅ Honest evaluation with PR-AUC as primary metric
- ✅ Full reproducibility (seeds, deterministic ops, saved splits)
- ✅ Notebook-driven experiments with saved artifacts

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
- **PR-AUC** (Primary metric for imbalanced data)
- **ROC-AUC**
- **F1 Score** (threshold selected on validation set)
- **Recall@K** (K ∈ {0.5%, 1%, 2%} of test size)

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
[Add citation information for Elliptic++ dataset]
```

---

## 🎯 Project Status

- [x] M1: Repository scaffold ✅
- [x] M2: Data loader & temporal splits ✅
- [ ] M3: GCN baseline notebook
- [ ] M4: GraphSAGE & GAT notebooks
- [ ] M5: Tabular baselines
- [ ] M6: Final verification & polish

---

## 📝 License

This project is for educational and demonstration purposes.

---

## 🤝 Contributing

This is a portfolio/demonstration project. Feel free to fork and adapt for your own use cases.

---

## 📧 Contact

**Author:** [Your Name]  
**GitHub:** [@BhaveshBytess](https://github.com/BhaveshBytess)

---

**⭐ If you find this project useful, please star the repository!**
