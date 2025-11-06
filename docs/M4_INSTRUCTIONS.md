# M4: GraphSAGE & GAT Training Instructions

## 🎯 Objective
Train GraphSAGE and GAT models on Kaggle GPU and compare with GCN baseline.

## 📋 Quick Steps

### 1. Upload Kaggle Notebook (2 minutes)
1. Go to https://www.kaggle.com/code
2. Click "New Notebook"
3. Settings → Accelerator → **GPU T4 x2**
4. Copy ALL content from `notebooks/04_graphsage_gat_kaggle.ipynb`
5. Paste into Kaggle

### 2. Link Dataset (1 minute)
1. Click "Add Data" (right sidebar)
2. Search "elliptic-fraud-detection" (your dataset)
3. Click "Add"

### 3. Run Training (25-30 minutes)
1. Click "Run All"
2. GraphSAGE trains first (~12-15 mins)
3. GAT trains second (~12-15 mins)
4. Results displayed automatically

### 4. Download Results (2 minutes)
Download these 4 files:
- `graphsage_metrics.json`
- `gat_metrics.json`
- `graphsage_best.pt`
- `gat_best.pt`

### 5. Save to Repository
```bash
# Place files:
reports/graphsage_metrics.json
reports/gat_metrics.json
checkpoints/graphsage_best.pt
checkpoints/gat_best.pt
```

---

## 📊 Expected Results

### GraphSAGE
**Hypothesis:** Better than GCN due to sampling
- Expected Test PR-AUC: 0.25-0.35 (vs GCN 0.20)
- Expected ROC-AUC: 0.78-0.82 (vs GCN 0.76)
- Less overfitting than GCN

### GAT
**Hypothesis:** Best of all three models
- Expected Test PR-AUC: 0.30-0.40
- Expected ROC-AUC: 0.80-0.85
- Attention learns important edges

---

## 🔍 What to Watch For

**During Training:**
- Both models should NOT have NaN issues (feature sanitization works)
- GraphSAGE typically converges faster
- GAT may fluctuate more early on (attention learning)

**Results:**
- Compare validation vs test gap (overfitting indicator)
- Check if PR-AUC improves over GCN (0.1976)
- Look at Recall@1% - should be higher

**Red Flags:**
- Val PR-AUC > 0.5 but Test PR-AUC < 0.2 → severe overfitting
- NaN warnings → report immediately

---

## 🎨 Model Differences

| Feature | GCN | GraphSAGE | GAT |
|---------|-----|-----------|-----|
| **Aggregation** | Mean | Sampled Mean | Weighted Attention |
| **Hidden Dims** | 128 | 128 | 64 (4 heads = 256) |
| **Learning Rate** | 0.001 | 0.001 | 0.005 |
| **Activation** | ReLU | ReLU | ELU |
| **Parameters** | ~24K | ~24K | ~48K |
| **Strengths** | Simple, fast | Scalable, generalizes | Learns edge importance |
| **Weaknesses** | Overfits | Still full-batch here | More parameters |

---

## 🧪 After Training

### Create Comparison Table
```python
import pandas as pd

results = {
    'Model': ['GCN', 'GraphSAGE', 'GAT'],
    'PR-AUC': [0.1976, <sage_pr_auc>, <gat_pr_auc>],
    'ROC-AUC': [0.7627, <sage_roc_auc>, <gat_roc_auc>],
    'F1': [0.2487, <sage_f1>, <gat_f1>],
    'Recall@1%': [0.0613, <sage_recall>, <gat_recall>]
}

df = pd.DataFrame(results)
print(df.to_markdown(index=False))
```

### Key Questions to Answer
1. **Which model performs best?** (highest PR-AUC)
2. **Which generalizes best?** (smallest val-test gap)
3. **Is attention helping?** (GAT vs GCN/SAGE)
4. **Are GNNs worth it?** (vs simpler models)

---

## 🐛 Troubleshooting

**GraphSAGE NaN:**
- Should not happen (same fixes as GCN)
- If it does: check data is linked correctly

**GAT NaN:**
- GAT is more sensitive to hyperparameters
- If NaN: try reducing heads to 2
- Or reduce hidden_channels to 32

**Out of Memory:**
- Reduce hidden_channels (128→64, 64→32)
- Reduce heads for GAT (4→2)
- Use single GPU in settings

**Slow Training:**
- Normal - GAT is ~2x slower than GCN
- GraphSAGE should be similar to GCN
- Total ~25-30 mins is expected

---

## ✅ Success Criteria

- [x] Both models train without NaN
- [x] GraphSAGE completes in ~15 mins
- [x] GAT completes in ~15 mins
- [x] All 4 files downloaded
- [x] Metrics look reasonable (not all zeros/NaN)

---

## 🚀 Next Steps After M4

1. **Update TASKS.md** - Mark M4 complete
2. **Commit results** to repository
3. **Create comparison analysis**
4. **Decide:** M5 (Tabular models) or M6 (Polish)

---

**Time Estimate:** ~35-40 minutes total (including download/upload)

Good luck! 🎯
