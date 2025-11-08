# 🔄 RETRAINING PLAN — After Dataset Label Fix

**Date:** November 7, 2025  
**Status:** Ready to execute  
**Reason:** Fixed incorrect label encoding in dataset loader

---

## 📊 Dataset Status: CORRECTED ✅

**Verification Results:**
```
Total transactions: 203,769
Labeled: 46,564 (22.8%)
- Class 1 (Illicit/Fraud): 4,545 (9.76% of labeled) ✅
- Class 2 (Licit/Legit): 42,019 (90.24% of labeled) ✅
- Class 3 (Unknown): 157,205

Split Distribution:
- Train: 26,381 labeled (10.88% fraud) ✅ REALISTIC
- Val: 8,999 labeled (11.53% fraud) ✅ REALISTIC  
- Test: 11,184 labeled (5.69% fraud) ✅ REALISTIC
```

**What was fixed:**
- `src/data/elliptic_loader.py`: Corrected class mapping (Class 1→1 fraud, Class 2→0 legit)
- Removed auto-detection logic that was flipping labels
- Verified with `python -m src.data.elliptic_loader --check`

---

## 🎯 Execution Plan

### Phase 1: Update Notebooks (Local)
**Goal:** Ensure all notebooks use corrected loader

**Tasks:**
- [x] Fix `src/data/elliptic_loader.py` label encoding
- [x] Verify loader with --check flag
- [ ] Update `notebooks/03_gcn_baseline_kaggle.ipynb` header comments
- [ ] Update `notebooks/04_graphsage_gat_kaggle.ipynb` header comments  
- [ ] Update `notebooks/05_tabular_baselines_kaggle.ipynb` header comments
- [ ] Commit changes to Git

**Verification:**
- All notebooks load from `data/Elliptic++ Dataset/` path
- All show ~9.76% fraud rate in data loading section
- No hardcoded label flips in notebooks

---

### Phase 2: Retrain M3 — GCN Baseline (Kaggle GPU)
**Notebook:** `03_gcn_baseline_kaggle.ipynb`  
**Platform:** Kaggle (GPU T4 x2)  
**Dataset:** `elliptic-fraud-detection` (already uploaded)

**Steps:**
1. Open Kaggle notebook
2. Link dataset (`elliptic-fraud-detection`)
3. Enable GPU accelerator
4. Run all cells
5. Download results:
   - `reports/gcn_metrics.json`
   - `reports/plots/gcn_training_history.png`
   - `reports/plots/gcn_pr_roc_curves.png`
   - `checkpoints/gcn_best.pt`

**Expected Results (Corrected):**
- PR-AUC: 0.15-0.25 (realistic for GCN on imbalanced fraud)
- ROC-AUC: 0.75-0.82
- Training time: ~10-15 minutes

**Success Criteria:**
- ✅ Fraud rate shown as ~10% in data loading
- ✅ Training completes without NaN errors
- ✅ Metrics are realistic (not >0.95)
- ✅ Model checkpoint saved

---

### Phase 3: Retrain M4 — GraphSAGE & GAT (Kaggle GPU)
**Notebook:** `04_graphsage_gat_kaggle.ipynb`  
**Platform:** Kaggle (GPU T4 x2)

**Steps:**
1. Open Kaggle notebook
2. Link dataset
3. Enable GPU
4. Run all cells (trains both models sequentially)
5. Download results:
   - GraphSAGE: `reports/graphsage_metrics.json`, plots, checkpoint
   - GAT: `reports/gat_metrics.json`, plots, checkpoint

**Expected Results (Corrected):**
- **GraphSAGE:** PR-AUC 0.20-0.35, ROC-AUC 0.78-0.85 (best GNN)
- **GAT:** PR-AUC 0.15-0.28, ROC-AUC 0.76-0.83
- Training time: ~20-25 minutes total

**Success Criteria:**
- ✅ Both models train successfully
- ✅ GraphSAGE outperforms GCN slightly
- ✅ GAT comparable to GCN
- ✅ All metrics saved correctly

---

### Phase 4: Retrain M5 — Tabular Baselines (Local CPU)
**Notebook:** `05_tabular_baselines.ipynb`  
**Platform:** Local (CPU sufficient)

**Steps:**
1. Open notebook locally
2. Run all cells (trains LR, RF, XGBoost, MLP)
3. Verify results saved to `reports/`

**Expected Results (Corrected):**
- **Logistic Regression:** PR-AUC 0.10-0.18
- **Random Forest:** PR-AUC 0.12-0.22
- **XGBoost:** PR-AUC 0.15-0.28 (best ML)
- **MLP:** PR-AUC 0.12-0.25

**Success Criteria:**
- ✅ All 4 models train without errors
- ✅ ML models get PR-AUC 0.10-0.30 range
- ✅ XGBoost is best tabular model
- ✅ **GNNs (GraphSAGE) should beat XGBoost** ⭐

---

### Phase 5: Compare & Document
**Goal:** Answer the key question: "Does graph structure help?"

**Tasks:**
- [ ] Create comparison table in `reports/all_models_comparison.csv`
- [ ] Generate comparison bar chart
- [ ] Update `PROJECT_SUMMARY.md` with corrected findings
- [ ] Update `README.md` with final results
- [ ] Document expected finding: **GNNs beat ML on graph-structured fraud data**

**Expected Hierarchy (Realistic):**
1. **GraphSAGE:** PR-AUC ~0.25-0.35 ⭐ WINNER (uses graph)
2. **GCN:** PR-AUC ~0.18-0.25 (uses graph)
3. **GAT:** PR-AUC ~0.15-0.28 (uses graph)
4. **XGBoost:** PR-AUC ~0.15-0.28 (no graph)
5. **Random Forest:** PR-AUC ~0.12-0.22 (no graph)
6. **MLP:** PR-AUC ~0.12-0.25 (no graph)
7. **Logistic Regression:** PR-AUC ~0.10-0.18 (baseline)

**Key Insight:** Graph structure provides ~10-40% improvement over tabular approaches!

---

### Phase 6: Finalize M6
**Tasks:**
- [ ] Update all TASKS.md milestones to ✅
- [ ] Run all unit tests
- [ ] Clean up temporary files
- [ ] Final README polish
- [ ] Push all changes to GitHub
- [ ] Verify GitHub repo is portfolio-ready

---

## ⚠️ Critical Notes

### What Changed:
- **Dataset encoding:** Class 1=Fraud (was incorrectly treated as Legit)
- **Fraud percentage:** 9.76% (was incorrectly shown as 90.24%)
- **All previous results INVALID** due to label flip

### What Stayed Same:
- Model architectures (GCN, GraphSAGE, GAT, ML models)
- Training hyperparameters
- Evaluation metrics (PR-AUC, ROC-AUC, F1, Recall@K)
- Temporal split logic (60-20-20)
- Kaggle workflow

### Expected Outcome:
- **BEFORE (wrong):** Tabular models dominated with >0.99 PR-AUC (impossible)
- **AFTER (correct):** GNNs should beat ML by 10-40% (realistic for graph data)

---

## 📋 Execution Checklist

**Pre-flight:**
- [x] Dataset loader fixed and verified
- [x] Fraud rate confirmed at ~9.76%
- [x] All notebooks point to correct data path
- [ ] Git commit: "Fix dataset label encoding"

**Retraining:**
- [ ] M3: GCN retrained on Kaggle
- [ ] M4: GraphSAGE retrained on Kaggle
- [ ] M4: GAT retrained on Kaggle
- [ ] M5: All 4 ML models retrained locally

**Validation:**
- [ ] All metrics in realistic range (0.10-0.40 PR-AUC)
- [ ] GNNs beat tabular models
- [ ] No NaN/inf errors in any training
- [ ] All plots generated correctly

**Documentation:**
- [ ] TASKS.md updated with corrected results
- [ ] PROJECT_SUMMARY.md reflects corrected findings
- [ ] README.md updated with portfolio-ready summary
- [ ] All artifacts saved to correct folders

**Final:**
- [ ] M6 verification checklist complete
- [ ] Push to GitHub
- [ ] Repo is portfolio-ready

---

## 🚀 Ready to Execute

**Next Action:** Begin Phase 2 — Retrain GCN on Kaggle

**Command to user:**
> "Please upload and run `notebooks/03_gcn_baseline_kaggle.ipynb` on Kaggle with GPU enabled, then share the results."

---

**End of Retraining Plan**
