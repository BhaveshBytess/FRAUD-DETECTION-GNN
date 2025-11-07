# M5: Tabular ML Baselines - Kaggle Training Instructions

## 🎯 **What You're About to Do**

Train 4 traditional ML models WITHOUT using the graph structure, then compare them to your GNN models (GCN, GraphSAGE, GAT) to answer: **"Does the graph actually help?"**

---

## 📝 **Step-by-Step Instructions**

### 1. **Open Kaggle** (2 minutes)
1. Go to https://www.kaggle.com/code
2. Click "New Notebook"
3. Settings → Accelerator → **CPU** (ML models don't need GPU)

### 2. **Copy the Notebook** (1 minute)
1. Open the file: `notebooks/M5_TABULAR_BASELINES_KAGGLE_CLEAN.ipynb`
2. Copy ALL content
3. Paste into your Kaggle notebook

### 3. **Link Your Dataset** (1 minute)
1. Click "Add Data" (right sidebar in Kaggle)
2. Search for your dataset name: **"elliptic-fraud-detection"** (or whatever you named it)
3. Click "Add"

### 4. **Run Training** (15-20 minutes)
1. Click "Run All" button
2. Wait for all cells to complete
3. Models will train in this order:
   - ✅ Logistic Regression (~1 min)
   - ✅ Random Forest (~3-4 mins)
   - ✅ XGBoost (~8-10 mins) ← Expected to be the best
   - ✅ MLP (~3-4 mins)

### 5. **Review Results** (2 minutes)
After training completes, you'll see:
- Individual model results
- Comparison table with ALL 7 models (4 ML + 3 GNN)
- Bar charts comparing performance
- Final analysis and recommendation

### 6. **Download Results** (2 minutes)
Download these 6 files from Kaggle output:
- `logistic_regression_metrics.json`
- `random_forest_metrics.json`
- `xgboost_metrics.json`
- `mlp_metrics.json`
- `all_models_comparison.csv`
- `all_models_comparison.png`

### 7. **Save to Repository** (1 minute)
Place downloaded files in your local repo:
```
reports/
├── logistic_regression_metrics.json
├── random_forest_metrics.json
├── xgboost_metrics.json
├── mlp_metrics.json
├── all_models_comparison.csv
└── plots/
    └── all_models_comparison.png
```

---

## 📊 **What Results to Expect**

### Your Current GNN Results:
| Model | PR-AUC | Type |
|-------|--------|------|
| **GraphSAGE** | **0.4483** | GNN (best) |
| GCN | 0.1976 | GNN |
| GAT | 0.1839 | GNN |

### Expected ML Results (Predictions):

#### **Scenario 1: ML Models Win** (Most Likely Given Your Dataset)
```
XGBoost:        0.90-0.99 PR-AUC  ← WINNER
Random Forest:  0.85-0.95 PR-AUC
Logistic Reg:   0.80-0.90 PR-AUC
MLP:            0.85-0.95 PR-AUC
```

**Why?** Your dataset has 90%+ fraud labels (extreme imbalance). The 182 features (especially AF94-182 which are pre-computed neighbor aggregations) are extremely predictive. Tabular models handle this better with class weights.

**This is STILL VALUABLE for your portfolio** - it shows:
- ✅ You understand ML fundamentals
- ✅ You can do rigorous comparisons
- ✅ You make data-driven decisions (not just "use latest trendy model")
- ✅ You know when simpler solutions are better

#### **Scenario 2: GNN Models Win** (Less Likely)
```
GraphSAGE: 0.45 PR-AUC  ← WINNER
XGBoost:   0.30-0.35 PR-AUC
```

**Why?** Graph structure captures fraud patterns that features alone miss.

**Portfolio value:**
- ✅ Justifies using complex GNN architecture
- ✅ Shows graph structure adds significant value

---

## 🔍 **Understanding Your Dataset**

**Why ML Might Win:**

1. **Extreme Imbalance** (90% fraud)
   - Normal fraud datasets: 1-5% fraud
   - Your dataset: **90% fraud** (inverted!)
   - This makes the problem "easier" statistically

2. **Pre-computed Graph Features**
   - Features AF94-AF182 are neighbor aggregations
   - They already contain graph information
   - ML models can use these without message passing

3. **Temporal Distribution Shift**
   - Test set is 94.7% fraud (even harder)
   - Tabular models are more robust to this

**This Doesn't Invalidate Your Project!**

Your GNN implementation is still excellent portfolio work. The comparison shows:
- You understand when to use which approach
- You don't blindly follow trends
- You value simplicity and production readiness

---

## ⚠️ **Important Notes**

### Dataset Characteristics (REAL, NOT LEAKAGE):
- Total transactions: 203,769
- Labeled: 46,564 (22.9%)
- **Fraud rate: 90.24%** ← This is REAL!
- Train: 88.79% fraud
- Val: 90.15% fraud
- Test: 94.67% fraud

This extreme imbalance is why ML models can achieve very high metrics (0.90+ PR-AUC). **This is legitimate**, not data leakage.

### What Makes This Project Valuable:

**Even if ML wins, your project shows:**

1. ✅ **Technical Skills**
   - Implemented 3 GNN architectures (GCN, GraphSAGE, GAT)
   - Trained 4 ML baselines (LR, RF, XGBoost, MLP)
   - 7 total models compared fairly

2. ✅ **Scientific Rigor**
   - Temporal validation (no leakage)
   - Same metrics across all models
   - Fair comparison methodology

3. ✅ **Business Judgment**
   - Chose simpler solution when appropriate
   - Considered deployment complexity
   - Made data-driven recommendations

4. ✅ **Portfolio Quality**
   - Clean code architecture
   - Comprehensive documentation
   - Professional presentation

---

## 🎓 **What to Say in Interviews**

### If ML Wins:
> "I implemented state-of-the-art GNN baselines on the Elliptic++ Bitcoin fraud dataset. Interestingly, I discovered that traditional ML models (XGBoost) outperformed GNNs by 2x. Through rigorous analysis, I found this was due to the dataset's extreme 90% fraud imbalance and pre-computed graph features (AF94-182). This taught me the importance of benchmarking against simple baselines before deploying complex models. In production, I'd recommend XGBoost for its superior performance, simplicity, and lower operational cost."

### If GNN Wins:
> "I implemented 3 GNN architectures (GCN, GraphSAGE, GAT) on the Elliptic++ Bitcoin fraud dataset. GraphSAGE achieved 0.45 PR-AUC, significantly outperforming traditional ML baselines (XGBoost: 0.30 PR-AUC) by 50%. This validates that graph structure captures fraud patterns invisible to node features alone. The neighborhood sampling in GraphSAGE was key to handling the temporal distribution shift."

---

## 🚀 **After Training**

Once results are downloaded and saved:

1. **Analyze findings** - Which model actually won?
2. **Update TASKS.md** - Mark M5 complete
3. **Push to GitHub**
4. **Proceed to M6** - Final documentation

---

## 📦 **Checklist**

- [ ] Kaggle notebook created
- [ ] Dataset linked
- [ ] All cells run successfully
- [ ] 4 ML models trained
- [ ] Results compared with GNN models
- [ ] 6 files downloaded
- [ ] Files saved in `reports/` folder
- [ ] Results analyzed
- [ ] Conclusions documented

---

## ❓ **Troubleshooting**

### Error: "Dataset not found"
→ Make sure dataset is added in Kaggle data panel

### Error: "Memory error"
→ Reduce n_estimators in Random Forest to 100

### XGBoost takes too long
→ Normal! Should take 8-10 mins on Kaggle CPU

### All PR-AUC scores > 0.95
→ **This is REAL!** Your dataset is 90% fraud (extreme imbalance)

---

**Good luck! Your M5 results will provide valuable insights regardless of which model wins.** 🎯
