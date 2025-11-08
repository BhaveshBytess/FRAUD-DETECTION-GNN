# 🔄 DATASET MIGRATION PLAN

## ✅ Current Status

**Verified:** Real Elliptic++ dataset is present in `data/Elliptic++ Dataset/`

**Fraud Rate:** 9.76% (REALISTIC!) ✅
- Total: 203,769 transactions
- Labeled: 46,564 (Class 1=legit: 42,019 | Class 2=fraud: 4,545)
- Unlabeled: 157,205 (Class 3)

---

## 🎯 Migration Goals

1. **Update data loader** to use correct dataset path and label mapping
2. **Fix label encoding** (Class 1→0=legit, Class 2→1=fraud, Class 3→unlabeled)
3. **Regenerate temporal splits** with correct fraud percentages
4. **Re-train all models** (GCN, GraphSAGE, GAT, tabular) with realistic data
5. **Update results** with honest performance metrics

---

## 📋 Step-by-Step Migration

### Phase 1: Data Loader Update (M2 Revision)

**Files to modify:**
- `src/data/elliptic_loader.py` - Update paths and label mapping

**Changes:**
```python
# Old path (if it was different):
# data_dir = "data/elliptic/"

# New path:
data_dir = "data/Elliptic++ Dataset/"

# Files:
features_file = "txs_features.csv"
classes_file = "txs_classes.csv" 
edges_file = "txs_edgelist.csv"

# Label mapping (CRITICAL FIX):
# Class 1 (legit) → 0
# Class 2 (fraud) → 1
# Class 3 (unknown) → -1 (exclude from training)
```

**Expected Output:**
- `splits.json` regenerated with ~9.76% fraud rate
- Temporal splits maintain similar distribution

---

### Phase 2: Verify Splits & Class Balance

**Run:**
```bash
python -m src.data.elliptic_loader --check
```

**Expected:**
- Train fraud %: ~9-10%
- Val fraud %: ~9-10%
- Test fraud %: ~9-10% (or lower due to temporal shift)
- NO 90% fraud rate!

---

### Phase 3: Re-train Models

#### M3: GCN Baseline
- [ ] Update `notebooks/03_gcn_baseline.ipynb`
- [ ] Re-run training (expect LOWER performance - realistic!)
- [ ] Expected PR-AUC: 0.15-0.30 (not 0.99!)
- [ ] Save new metrics & plots

#### M4: GraphSAGE & GAT
- [ ] Update `notebooks/04_graphsage_gat_kaggle.ipynb`
- [ ] Re-train both models
- [ ] Expected: GraphSAGE > GCN (but realistic scores)

#### M5: Tabular Baselines
- [ ] Update `notebooks/05_tabular_baselines.ipynb`
- [ ] Re-train LR, RF, XGBoost, MLP
- [ ] Expected PR-AUC: 0.20-0.35 (realistic!)

---

### Phase 4: Results Comparison & Analysis

**Expected Hierarchy (REALISTIC):**
```
XGBoost:      0.25-0.35 PR-AUC (best tabular)
GraphSAGE:    0.30-0.45 PR-AUC (best GNN - may win!)
Random Forest: 0.20-0.30 PR-AUC
GCN:          0.15-0.25 PR-AUC
GAT:          0.15-0.25 PR-AUC
Logistic Reg: 0.10-0.20 PR-AUC
```

**Key Question:** Does graph structure help?
- If GraphSAGE > XGBoost → **YES, graph helps!** ✅
- If XGBoost > GraphSAGE → **Features alone sufficient**

---

### Phase 5: Update Documentation

- [ ] Update `TASKS.md` with corrected results
- [ ] Update `README.md` with realistic findings
- [ ] Create `reports/FINAL_RESULTS.md`
- [ ] Update `PROJECT_SUMMARY.md`

---

## 🚨 What Changed from Previous Run

| Aspect | OLD (Wrong) | NEW (Correct) |
|--------|------------|---------------|
| Dataset path | `data/elliptic/` | `data/Elliptic++ Dataset/` |
| Fraud rate | 90.24% ❌ | 9.76% ✅ |
| Label encoding | FLIPPED | Class 1→0, Class 2→1 |
| XGBoost PR-AUC | 0.99 (impossible) | 0.25-0.35 (realistic) |
| GraphSAGE PR-AUC | 0.45 (suspicious) | 0.30-0.45 (realistic) |
| Conclusion | "Tabular dominates" ❌ | TBD - need real comparison ✅ |

---

## ⏱️ Estimated Timeline

- Phase 1 (Data loader): **15 min**
- Phase 2 (Verify splits): **5 min**
- Phase 3 (Re-train all): **2-3 hours** (with GPU)
- Phase 4 (Analysis): **30 min**
- Phase 5 (Docs): **30 min**

**Total:** ~4 hours

---

## 🎯 Success Criteria

✅ All models trained on real dataset with realistic fraud rate (~10%)
✅ Performance metrics are believable (PR-AUC: 0.15-0.45 range)
✅ Honest comparison between GNN vs tabular approaches
✅ Clear answer to: "Does graph structure help for fraud detection?"
✅ All artifacts (plots, metrics, checkpoints) updated
✅ Documentation reflects true findings

---

**Status:** Ready to begin Phase 1 ✅

