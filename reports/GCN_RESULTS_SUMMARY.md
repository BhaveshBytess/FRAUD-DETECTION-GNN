# GCN Baseline Results - Elliptic++ Fraud Detection

**Date:** November 6, 2025  
**Model:** Graph Convolutional Network (GCN)  
**Dataset:** Elliptic++ (Full - 203,769 transactions)  
**Training Platform:** Kaggle GPU T4 x2

---

## 📊 Performance Summary

### Test Set Metrics

| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| **PR-AUC** | 0.1976 | >0.60 | ⚠️ Below |
| **ROC-AUC** | 0.7627 | >0.80 | 🟡 Close |
| **F1 Score** | 0.2487 | >0.30 | ⚠️ Below |
| **Recall@0.5%** | 0.0393 | - | - |
| **Recall@1.0%** | 0.0613 | >0.40 | ⚠️ Below |
| **Recall@2.0%** | 0.1053 | - | - |

### Training Details

- **Best Validation PR-AUC:** 0.5710
- **Best Epoch:** 100
- **Threshold (F1-optimized):** 0.7148
- **Training Time:** ~15 minutes on GPU
- **Parameters:** 23,682 (2-layer GCN, 128 hidden)

---

## 📈 Data Statistics

### Full Dataset
- **Total Nodes:** 203,769 transactions
- **Total Edges:** 234,355 flows + 203,769 self-loops
- **Features:** 182 per node
- **Labeled:** 46,564 (22.9%)
  - Fraud: 4,545 (9.8% of labeled)
  - Legitimate: 42,019 (90.2% of labeled)
- **Unlabeled:** 157,205 (77.1%)

### Split Distribution

| Split | Nodes | Fraud | % Fraud | Time Range |
|-------|-------|-------|---------|------------|
| Train | 26,381 | 2,871 | 10.88% | ≤ 29 |
| Val | 8,999 | 1,038 | 11.53% | 30-39 |
| Test | 11,184 | 636 | **5.69%** | > 39 |

**Key Observation:** Fraud rate drops significantly in test set (temporal drift)

---

## 🔍 Analysis

### What Worked ✅

1. **GPU Training Successful**
   - No NaN issues after feature sanitization
   - Full-batch training on 203K nodes
   - Converged in 100 epochs

2. **Decent ROC-AUC**
   - 0.76 indicates good ranking ability
   - Model can separate positive/negative classes

3. **Infrastructure Complete**
   - Data pipeline validated
   - End-to-end training works
   - All artifacts generated

### Issues Identified ⚠️

1. **Severe Overfitting**
   - Val PR-AUC: 0.57
   - Test PR-AUC: 0.20
   - **2.9x performance drop**

2. **Temporal Distribution Shift**
   - Train/Val fraud rate: ~11%
   - Test fraud rate: 5.7%
   - **Model doesn't generalize to future**

3. **Low Precision-Recall Performance**
   - PR-AUC of 0.20 is only slightly better than random (0.057 baseline)
   - Struggles with severe class imbalance

4. **Poor Top-K Recall**
   - Only 6% of fraud caught in top 1% predictions
   - Not operationally useful for fraud detection

---

## 💡 Insights

### Why Test Performance is Lower

1. **Temporal Drift**
   - Bitcoin fraud patterns evolve
   - Later periods have different characteristics
   - GCN memorizes training graph structure

2. **Class Imbalance Worsens**
   - Fraud rate halves in test period
   - Model biased toward majority class
   - Threshold optimization on val doesn't transfer

3. **Graph Structure Changes**
   - New nodes/edges in test period
   - Different connectivity patterns
   - GCN relies heavily on graph topology

### Model Behavior

- **Good at ranking** (ROC-AUC 0.76)
- **Poor at precision-recall trade-off** (PR-AUC 0.20)
- **Conservative predictions** (high threshold 0.71)
- **Misses most fraud** (Recall@1% only 6%)

---

## 🎯 Recommendations

### Immediate Actions

1. **Try GraphSAGE**
   - Sampling-based, better generalization
   - Less prone to overfitting
   - Works well on evolving graphs

2. **Address Class Imbalance**
   - Focal loss instead of cross-entropy
   - Class weights (inverse frequency)
   - Oversample minority class

3. **Temporal Validation**
   - Multiple time-based splits
   - Rolling window validation
   - Evaluate on different test periods

### Model Improvements

1. **Architecture:**
   - Add skip connections (ResGCN)
   - Batch normalization
   - Reduce capacity (prevent overfitting)

2. **Training:**
   - Lower learning rate (0.0001)
   - More regularization (dropout 0.5)
   - Early stopping on test PR-AUC

3. **Features:**
   - Temporal features (time since first tx)
   - Graph statistics (degree, centrality)
   - Transaction amount patterns

### Ensemble Strategy

Combine multiple approaches:
- **GNN models:** GCN + GraphSAGE + GAT
- **Tabular models:** XGBoost, Random Forest
- **MLP on node features**
- **Weighted voting or stacking**

---

## 📁 Artifacts

All results saved to repository:

```
reports/
├── gcn_metrics.json           # Test set metrics
└── plots/
    ├── gcn_training_history.png
    └── gcn_pr_roc_curves.png

checkpoints/
└── gcn_best.pt                # Trained model (epoch 100)
```

---

## 🔬 Technical Details

### Model Architecture
```
GCN(
  (convs): ModuleList(
    (0): GCNConv(182, 128)
    (1): GCNConv(128, 2)
  )
)
Parameters: 23,682
```

### Hyperparameters
- **Learning rate:** 0.001
- **Weight decay:** 0.0005
- **Dropout:** 0.4
- **Optimizer:** Adam
- **Loss:** CrossEntropyLoss
- **Early stopping:** Patience 15, metric PR-AUC

### Data Preprocessing
- Z-score normalization (robust)
- Inf/NaN replacement with 0.0
- Manual self-loops added
- Temporal splits (60/20/20)

---

## ✅ Conclusion

**GCN Baseline Established**

While performance is below targets, this provides a critical baseline:

✅ **Infrastructure validated** - data pipeline works end-to-end  
✅ **Benchmark established** - future models can compare  
✅ **Issues identified** - clear path for improvement  
✅ **Full dataset processed** - no sampling artifacts  

**Next:** Implement GraphSAGE and GAT models to compare approaches.

---

**Milestone M3: COMPLETE** ✅
