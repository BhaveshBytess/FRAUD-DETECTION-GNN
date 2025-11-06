# GCN Training on Kaggle - Instructions

## 🎯 Objective
Train the GCN model on Kaggle's free GPU and save results back to local repository.

## 📋 Steps to Run on Kaggle

### 1. Upload Dataset to Kaggle
1. Go to https://www.kaggle.com/datasets
2. Click "New Dataset"
3. Upload these files from `Elliptic++ Dataset/`:
   - `txs_features.csv`
   - `txs_classes.csv`
   - `txs_edgelist.csv`
4. Name it: "elliptic-fraud-detection"
5. Make it private/public as needed

### 2. Create New Kaggle Notebook
1. Go to https://www.kaggle.com/code
2. Click "New Notebook"
3. Settings → Accelerator → **GPU T4 x2** (free)
4. Copy the code from `notebooks/03_gcn_baseline_kaggle.ipynb`

### 3. Link Dataset
1. In Kaggle notebook, click "Add Data"
2. Search for your "elliptic-fraud-detection" dataset
3. Click "Add"

### 4. Run the Notebook
1. Click "Run All" or run cell by cell
2. Training will take ~10-15 minutes on GPU
3. Results will be generated

### 5. Download Results
After successful run, download these files:
- `gcn_metrics.json` - Test set metrics
- `gcn_training_history.png` - Training curves
- `gcn_pr_roc_curves.png` - PR/ROC curves
- `gcn_best.pt` - Model checkpoint

### 6. Save to Local Repository
Place downloaded files in:
```
reports/gcn_metrics.json
reports/plots/gcn_training_history.png
reports/plots/gcn_pr_roc_curves.png
checkpoints/gcn_best.pt
```

## 📊 Expected Results

**Training:**
- ~30-50 epochs before early stopping
- Best validation PR-AUC: ~0.65-0.75 (expected)
- Training time: ~10-15 minutes on GPU

**Test Metrics:**
- PR-AUC (primary): Target >0.60
- ROC-AUC: Target >0.80
- F1 Score: Target >0.30
- Recall@1%: Target >0.40

## 🐛 Troubleshooting

**If dataset not found:**
- Check dataset path in code
- Ensure dataset is linked to notebook

**If CUDA out of memory:**
- Reduce batch size (not applicable for full-batch)
- Reduce hidden_channels to 64
- Use single GPU in settings

**If kernel crashes:**
- Restart kernel
- Check data loading completed
- Reduce model size

## ✅ Verification Checklist

Before downloading results:
- [ ] Training completed without errors
- [ ] Metrics are reasonable (not NaN or 0)
- [ ] Plots generated successfully
- [ ] Best model saved

## 🔄 Alternative: Google Colab

If Kaggle doesn't work, try Google Colab:
1. Upload notebook to Colab
2. Runtime → Change runtime type → GPU
3. Upload dataset files or mount Google Drive
4. Run notebook

---

**Note:** The Kaggle notebook is self-contained and includes all necessary code. Just upload dataset, run, and download results!
