# When Graph Neural Networks Fail: Revisiting Graph Learning on the Elliptic++ Dataset

**TL;DR.** XGBoost (tabular) outperforms standard GNN baselines (GraphSAGE, GCN, GAT) on the Elliptic++ Bitcoin transaction dataset. Feature-ablation shows precomputed neighbor aggregates (AF94–AF182) explain much of the gap.

## Abstract

We implement reproducible baselines (GCN, GraphSAGE, GAT) and tabular models (XGBoost, RandomForest, MLP) using strict temporal splits on Elliptic++. Main finding: **XGBoost (PR-AUC 0.669)** vs **GraphSAGE (PR-AUC 0.448)**. Ablations reveal AF94–AF182 encode neighbor aggregates; removing them improves GraphSAGE to ~0.556 PR-AUC.

## Results snapshot

| Model | PR-AUC | ROC-AUC | F1 |
|---|---:|---:|---:|
| XGBoost | **0.669** | 0.888 | 0.699 |
| Random Forest | 0.658 | 0.877 | 0.694 |
| GraphSAGE | 0.448 | 0.821 | 0.453 |
| GCN | 0.198 | 0.763 | 0.249 |
| GAT | 0.184 | 0.794 | 0.290 |

![Model Comparison](reports/plots/all_models_comparison.png)

*(Full results & figures: `reports/` — see `docs/M5_RESULTS_SUMMARY.md`.)*

## Quickstart (minimal)

```bash
# 1) Create env & install
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2) Place dataset (NOT included) at:
#    data/Elliptic++ Dataset/txs_features.csv
#    data/Elliptic++ Dataset/txs_classes.csv
#    data/Elliptic++ Dataset/txs_edgelist.csv
python -m src.data.elliptic_loader --root "data/Elliptic++ Dataset" --check

# 3) Train GraphSAGE (example)
python -m src.train --config configs/graphsage.yaml

# 4) Run tabular baselines
python scripts/run_m5_tabular.py --config configs/m5_xgboost.yaml
```

## Dataset

**Elliptic++ Bitcoin transaction graph** (approx. 203,769 nodes; 234,355 edges; 182 features). 

⚠️ **Data is NOT included in this repository.**

Download from: [Kaggle Elliptic++ Dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)

Place the following files in `data/Elliptic++ Dataset/`:
- `txs_features.csv`
- `txs_classes.csv`
- `txs_edgelist.csv`

Please respect dataset licensing terms.

## Where to read more

- **Full documentation:** [`docs/README_FULL.md`](docs/README_FULL.md) — Complete technical README with all sections
- **Project report:** [`PROJECT_REPORT.md`](PROJECT_REPORT.md) — Detailed analysis and findings
- **Milestones & results:**
  - `docs/M5_RESULTS_SUMMARY.md` — Tabular baselines results
  - `docs/M7_RESULTS.md` — Feature ablation experiments
  - `docs/M8_INTERPRETABILITY.md` — SHAP & saliency analysis
  - `docs/M9_TEMPORAL.md` — Temporal robustness study
- **Artifacts:** `reports/` — Metrics (JSON/CSV) and plots
- **Notebooks:** `notebooks/` — Exploratory analysis (run 03→08)

## Cite & contact

**Citation metadata:** See [`CITATION.cff`](CITATION.cff)

**BibTeX:**
```bibtex
@software{elliptic_gnn_baselines_2025,
  title = {When Graph Neural Networks Fail: Revisiting Graph Learning on the Elliptic++ Dataset},
  author = {Bytes, Bhavesh},
  year = {2025},
  url = {https://github.com/BhaveshBytess/FRAUD-DETECTION-GNN},
  license = {MIT}
}
```

**License:** MIT

**Contact:** Bhavesh Bytes — [GitHub: @BhaveshBytess](https://github.com/BhaveshBytess)

---

**Project Status:** ✅ Complete (Milestones M1–M9)  
**Last Updated:** November 2025
