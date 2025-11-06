# M5: Tabular Baselines Training Instructions

## 🎯 Objective
Train traditional ML models (no graph) and compare with GNN models to answer: **"Does the graph structure actually help?"**

---

## 📋 Quick Steps

### 1. Upload Kaggle Notebook (2 minutes)
1. Go to https://www.kaggle.com/code
2. Click "New Notebook"
3. Settings → Accelerator → **CPU** (ML models don't need GPU)
4. Copy ALL content from `notebooks/05_tabular_baselines_kaggle.ipynb`
5. Paste into Kaggle

### 2. Link Dataset (1 minute)
1. Click "Add Data" (right sidebar)
2. Search "elliptic-fraud-detection" (your dataset)
3. Click "Add"

### 3. Run Training (15-20 minutes)
1. Click "Run All"
2. Models train in order:
   - Logistic Regression (~1 min)
   - Random Forest (~3-4 mins)
   - XGBoost (~8-10 mins) ← Expected best
   - MLP (~3-4 mins)
3. Results displayed automatically

### 4. Download Results (2 minutes)
Download these 5 files:
- `logistic_regression_metrics.json`
- `random_forest_metrics.json`
- `xgboost_metrics.json`
- `mlp_metrics.json`
- `all_models_comparison.csv`
- `all_models_comparison.png` (visualization)

### 5. Save to Repository
```bash
# Place files:
reports/logistic_regression_metrics.json
reports/random_forest_metrics.json
reports/xgboost_metrics.json
reports/mlp_metrics.json
reports/all_models_comparison.csv
reports/plots/all_models_comparison.png
```

---

## 📊 **Expected Results**

### Scenario 1: Graph Is Essential
```
XGBoost PR-AUC:    0.25-0.30
GraphSAGE PR-AUC:  0.45  ✅

→ Graph structure adds 50%+ improvement!
→ GNNs are justified
→ Recommendation: Deploy GraphSAGE
```

### Scenario 2: Graph Helps Moderately
```
XGBoost PR-AUC:    0.35-0.42
GraphSAGE PR-AUC:  0.45  ✅

→ Graph adds some value (~10-20%)
→ GNNs are worth it but not essential
→ Recommendation: GraphSAGE for best results, XGBoost for simplicity
```

### Scenario 3: Graph Is Not Needed (Unlikely)
```
XGBoost PR-AUC:    0.48-0.55  ✅
GraphSAGE PR-AUC:  0.45

→ Features already encode graph info (AF94-182)
→ Simpler model performs better
→ Recommendation: Deploy XGBoost (easier to maintain)
```

---

## 🔍 **What to Watch For**

### During Training

**Logistic Regression:**
- Fastest (~1 min)
- Linear baseline
- Expected PR-AUC: 0.15-0.25

**Random Forest:**
- Moderate speed (~3-4 mins)
- Non-linear patterns
- Expected PR-AUC: 0.25-0.35

**XGBoost:**
- Slowest (~8-10 mins)
- **Expected best ML model**
- Expected PR-AUC: 0.30-0.45
- Uses early stopping (watch validation score)

**MLP:**
- Neural network without graph
- Expected PR-AUC: 0.20-0.35
- Should be worse than XGBoost

### Red Flags
- ⚠️ XGBoost PR-AUC < 0.20 → Something wrong with data
- ⚠️ All ML models PR-AUC > 0.50 → Check for data leakage
- ⚠️ Class weight warnings → Expected (5.7% fraud is very imbalanced)

---

## 🎨 **Model Comparison**

| Model | Type | Uses Graph? | Expected PR-AUC | Complexity |
|-------|------|-------------|-----------------|------------|
| **Logistic Regression** | Linear | ❌ | 0.15-0.25 | Very Low |
| **Random Forest** | Tree Ensemble | ❌ | 0.25-0.35 | Medium |
| **XGBoost** | Gradient Boosting | ❌ | 0.30-0.45 | Medium |
| **MLP** | Neural Net | ❌ | 0.20-0.35 | High |
| **GCN** | GNN | ✅ | 0.20 | High |
| **GAT** | GNN | ✅ | 0.18 | Very High |
| **GraphSAGE** | GNN | ✅ | **0.45** | High |

**Key Question:** Does XGBoost beat GraphSAGE?

---

## 🧪 **After Training**

### Key Questions to Answer

1. **What's the best ML model?**
   - Usually XGBoost
   - Compare PR-AUC across all 4 ML models

2. **Does the graph help?**
   - Compare: Best ML vs GraphSAGE
   - If GraphSAGE wins by >10% → Graph is valuable
   - If gap is <5% → Graph doesn't add much

3. **Which model to deploy?**
   - Highest PR-AUC overall
   - Balance: performance vs operational complexity
   - GraphSAGE: Better performance, needs PyG
   - XGBoost: Simpler deployment, sklearn only

4. **Feature insights?**
   - Check XGBoost/RF feature importance
   - Which features matter most?
   - Do aggregated features (AF94-182) dominate?

---

## 📈 **Analysis Template**

After getting results, analyze:

```python
# Best ML model
best_ml = max(lr, rf, xgb, mlp) by PR-AUC

# Gap analysis
gap = (GraphSAGE_PR_AUC - best_ml_PR_AUC) / best_ml_PR_AUC * 100

if gap > 20%:
    print("🏆 Graph is ESSENTIAL! GNNs win decisively.")
elif gap > 5%:
    print("✅ Graph helps moderately. GNNs worth it.")
elif gap > -5%:
    print("⚖️ Roughly equal. Choose based on ops complexity.")
else:
    print("⚠️ ML wins! Graph adds noise or features sufficient.")
```

---

## 🐛 **Troubleshooting**

### Logistic Regression Issues
- **Error:** Memory error → Reduce max_iter to 500
- **Low score:** Expected, it's a simple linear model

### Random Forest Issues
- **Slow:** Reduce n_estimators to 50
- **Memory error:** Reduce max_depth to 10

### XGBoost Issues
- **Error: "Invalid feature names"** → Check feature columns
- **Slow:** Normal for 200 trees, ~8-10 mins
- **Early stopping at epoch 5:** Model converged fast (good sign)

### MLP Issues
- **ConvergenceWarning:** Increase max_iter to 200
- **Poor performance:** Normal, MLP without graph isn't great

---

## ✅ **Success Criteria**

- [x] All 4 ML models train without errors
- [x] XGBoost completes in ~10 mins
- [x] All models have PR-AUC > 0.10 (sanity check)
- [x] Comparison table shows clear ranking
- [x] Visualization created
- [x] All 6 files downloaded

---

## 🚀 **Next Steps After M5**

1. **Analyze results** - Which model wins?
2. **Update TASKS.md** - Mark M5 complete
3. **Create final summary** - Document key findings
4. **Proceed to M6** - Final polish & documentation

---

## 💡 **Expected Insights**

### If XGBoost PR-AUC ≈ 0.30-0.35
*"GraphSAGE (0.45) significantly outperforms best ML model (0.32) by 40%. Graph structure is crucial for fraud detection on Elliptic++. The transaction network captures fraud patterns that individual features miss."*

### If XGBoost PR-AUC ≈ 0.40-0.45
*"GraphSAGE (0.45) marginally outperforms best ML model (0.42) by 7%. While graph helps, pre-computed aggregated features (AF94-182) already encode much of the graph structure. Both approaches viable."*

### If XGBoost PR-AUC > 0.50
*"XGBoost (0.52) outperforms GraphSAGE (0.45) by 15%. The graph adds noise rather than signal. Aggregated features are sufficient. Recommend deploying XGBoost for simplicity and performance."*

---

## 📊 **Portfolio Impact**

**With M5, you can say:**

✅ "Implemented comprehensive ML pipeline"  
✅ "Compared 7 different models (3 GNN + 4 ML)"  
✅ "Rigorous scientific evaluation"  
✅ "Justified architecture choice with data"  
✅ "Understand when GNNs add value vs traditional ML"  

**This is production-level ML engineering!** 🎯

---

**Time Estimate:** ~25-30 minutes total

Good luck! 🚀
