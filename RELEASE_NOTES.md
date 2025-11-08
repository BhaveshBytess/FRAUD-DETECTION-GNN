# v1.0.0: Elliptic++ Fraud Detection - When Do GNNs Add Value?

## 🎯 Release Summary

This release marks the completion of a comprehensive comparative study investigating **when Graph Neural Networks provide marginal value over tabular ML models** for fraud detection on the Elliptic++ Bitcoin transaction dataset.

**Main Finding:** Features already encode graph aggregations — GNNs redundant unless features are raw.

---

## 🔬 Key Research Contributions

### 1. Feature Dominance Hypothesis — CONFIRMED ✅

Through ablation experiments (M7), we confirmed that tabular features AF94–AF182 already encode neighbor-aggregated information:

| Model | Config | PR-AUC | Δ vs Full | Interpretation |
|-------|--------|--------|-----------|----------------|
| **XGBoost** | Full (AF1–182) | 0.669 | — | Baseline |
| XGBoost | Local only (AF1–93) | 0.648 | **−3%** | **Barely affected** |
| **GraphSAGE** | Full (AF1–182) | 0.448 | — | Redundant encoding |
| GraphSAGE | Local only (AF1–93) | **0.556** | **+24%** | **GNN unlocked!** |

**Evidence:**
- Neighbor averages correlate **r=0.74–0.89** with aggregate features
- Manual graph metrics correlate **r=0.63–0.65** with aggregates
- AF94–AF182 are literally pre-computed neighbor aggregations

### 2. Interpretability Analysis (M8)

- **XGBoost (full):** Heavily relies on aggregate features (SHAP analysis)
- **GraphSAGE (local-only):** Learns from graph structure via message passing
- Models learn *different* representations: pre-computed features vs dynamic aggregation

### 3. Temporal Robustness (M9)

- **XGBoost:** Stable 0.67–0.78 PR-AUC across time shifts
- **GraphSAGE (local):** Improves 0.41 → 0.56 with earlier training windows (+35%)
- **Finding:** GNNs handle temporal drift better when trained on raw features

---

## 📊 Complete Results

**Model Performance Rankings (PR-AUC):**

| Rank | Model | Type | PR-AUC | ROC-AUC | F1 | Recall@1% |
|------|-------|------|--------|---------|----|-----------| 
| 🥇 1 | **XGBoost** | Tabular | **0.669** | 0.888 | 0.699 | 0.175 |
| 🥈 2 | Random Forest | Tabular | 0.658 | 0.877 | 0.694 | 0.175 |
| 🥉 3 | **GraphSAGE** | GNN | **0.448** | 0.821 | 0.453 | 0.148 |
| 4 | MLP | Tabular | 0.364 | 0.830 | 0.486 | 0.094 |
| 5 | GCN | GNN | 0.198 | 0.763 | 0.249 | 0.061 |
| 6 | GAT | GNN | 0.184 | 0.794 | 0.290 | 0.013 |
| 7 | Logistic Regression | Tabular | 0.164 | 0.824 | 0.256 | 0.005 |

---

## 🎓 Milestones Completed (10/10)

- ✅ **M1:** Repository scaffold & infrastructure
- ✅ **M2:** Dataset loader with temporal splits
- ✅ **M3:** GCN baseline implementation
- ✅ **M4:** GraphSAGE & GAT models
- ✅ **M5:** Tabular baselines (LR, RF, XGBoost, MLP)
- ✅ **M6:** Documentation & comparative analysis
- ✅ **M7:** Causality & Feature Dominance — **HYPOTHESIS CONFIRMED**
- ✅ **M8:** Interpretability (SHAP + GNN saliency)
- ✅ **M9:** Temporal Robustness Study
- ✅ **M10:** Final polish & release preparation

---

## 📁 What's Included

### Documentation
- **`PROJECT_REPORT.md`** — Publication-style comprehensive report (13KB)
- **`README.md`** — Complete project overview with findings
- **`PROJECT_SUMMARY.md`** — Detailed narrative and evidence
- **`docs/M7_RESULTS.md`** — Feature dominance ablation results
- **`docs/M8_INTERPRETABILITY.md`** — SHAP + GNN saliency analysis
- **`docs/M9_TEMPORAL.md`** — Temporal robustness findings
- **`LICENSE`** — MIT License

### Code & Notebooks
- 8 reproducible notebooks (`notebooks/03-08_*.ipynb`)
- 12+ training scripts (`scripts/run_m*.py`)
- Model implementations (GCN, GraphSAGE, GAT)
- Complete data pipeline

### Artifacts
- **25+ result files** in `reports/`
- M7 ablation experiments (7 CSV/JSON files)
- M8 interpretability (3 CSV/JSON + plots)
- M9 temporal robustness (1 CSV file)
- All baseline model metrics

---

## 💡 Key Takeaways

> **Graph structure is valuable — but dataset features already captured it through pre-computed aggregations.**

**For Practitioners:**
- ✅ Use **XGBoost** when features include graph aggregations (fast, interpretable, CPU-friendly)
- ✅ Use **GNNs** when features are raw and graph structure is critical
- ✅ Always check feature-structure redundancy before selecting models

**For Researchers:**
- Feature engineering quality matters more than model architecture
- Ablation studies reveal hidden feature-structure redundancies
- Correlation analysis essential for understanding feature provenance
- Temporal robustness tests reveal model generalization characteristics

---

## 🚀 Getting Started

```bash
# Clone repository
git clone https://github.com/BhaveshBytess/FRAUD-DETECTION-GNN.git
cd FRAUD-DETECTION-GNN

# Install dependencies
pip install -r requirements.txt

# Explore results
jupyter notebook notebooks/

# Read full report
cat PROJECT_REPORT.md
```

---

## 📖 Citation

If you use this work, please cite:

```bibtex
@misc{elliptic-gnn-2025,
  title={Elliptic++ Fraud Detection: When Do Graph Neural Networks Add Value?},
  author={Bhavesh Bytes},
  year={2025},
  url={https://github.com/BhaveshBytess/FRAUD-DETECTION-GNN},
  note={v1.0.0}
}
```

---

## 🎉 Acknowledgments

- Elliptic++ dataset providers
- PyTorch Geometric community
- Kaggle for GPU resources

---

**Project Status:** ✅ Complete  
**Release Date:** 2025-11-08  
**License:** MIT

---

**⭐ If you find this project useful, please star the repository!**
