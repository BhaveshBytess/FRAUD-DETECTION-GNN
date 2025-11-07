"""
Retrain all models with corrected Elliptic++ dataset.

This script retrains all GNN and tabular models after the dataset encoding fix.
Previous results used flipped labels (90% fraud) and are invalid.

Expected performance with corrected data:
- Fraud rate: ~11% in train/val, ~6% in test
- PR-AUC: 0.15-0.30 (realistic for imbalanced fraud detection)
- GNNs should outperform tabular models (graph structure helps!)
"""
import sys
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from src.data import EllipticDataset
from src.models.gcn import GCN, GCNTrainer
from src.models.graphsage import GraphSAGE, GraphSAGETrainer
from src.models.gat import GAT, GATTrainer
from src.utils.seed import set_all_seeds
from src.utils.metrics import compute_metrics, find_best_f1_threshold
from src.utils.logger import append_metrics_to_csv

# Reproducibility
SEED = 42
set_all_seeds(SEED)

print("=" * 80)
print(" RETRAINING ALL MODELS - CORRECTED DATASET")
print("=" * 80)
print(f"\nDataset fix: Class 1=Fraud (90% of labeled), Class 2=Legit (10%)")
print(f"After temporal split: ~11% fraud in train/val, ~6% in test")
print(f"\nSeed: {SEED}")
print("=" * 80)

# Load dataset
print("\n[1/6] Loading dataset...")
dataset = EllipticDataset(root='data/Elliptic++ Dataset')
data = dataset.load(verbose=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\n[*] Device: {device}")

# Prepare data for tabular models
print("\n[2/6] Preparing tabular data...")
X_train = data.x[data.train_mask].numpy()
y_train = data.y[data.train_mask].numpy()
X_val = data.x[data.val_mask].numpy()
y_val = data.y[data.val_mask].numpy()
X_test = data.x[data.test_mask].numpy()
y_test = data.y[data.test_mask].numpy()

print(f"   Train: {X_train.shape}, Fraud: {y_train.mean()*100:.2f}%")
print(f"   Val:   {X_val.shape}, Fraud: {y_val.mean()*100:.2f}%")
print(f"   Test:  {X_test.shape}, Fraud: {y_test.mean()*100:.2f}%")

results = []

# ========================================
# TABULAR MODELS
# ========================================

print("\n" + "=" * 80)
print(" TABULAR BASELINES (No Graph)")
print("=" * 80)

# Logistic Regression
print("\n[3/6] Training Logistic Regression...")
start = time.time()
lr_model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=SEED
)
lr_model.fit(X_train, y_train)
lr_probs = lr_model.predict_proba(X_test)[:, 1]
lr_metrics = compute_metrics(y_test, lr_probs)
print(f"   Time: {time.time()-start:.1f}s | PR-AUC: {lr_metrics['pr_auc']:.4f} | ROC-AUC: {lr_metrics['roc_auc']:.4f}")
results.append(('Logistic Regression', 'Tabular', lr_metrics))

# Random Forest
print("\n[4/6] Training Random Forest...")
start = time.time()
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',
    random_state=SEED,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_probs = rf_model.predict_proba(X_test)[:, 1]
rf_metrics = compute_metrics(y_test, rf_probs)
print(f"   Time: {time.time()-start:.1f}s | PR-AUC: {rf_metrics['pr_auc']:.4f} | ROC-AUC: {rf_metrics['roc_auc']:.4f}")
results.append(('Random Forest', 'Tabular', rf_metrics))

# XGBoost
print("\n[5/6] Training XGBoost...")
start = time.time()
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=SEED,
    eval_metric='aucpr',
    early_stopping_rounds=10
)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
xgb_metrics = compute_metrics(y_test, xgb_probs)
print(f"   Time: {time.time()-start:.1f}s | PR-AUC: {xgb_metrics['pr_auc']:.4f} | ROC-AUC: {xgb_metrics['roc_auc']:.4f}")
results.append(('XGBoost', 'Tabular', xgb_metrics))

# ========================================
# GNN MODELS
# ========================================

print("\n" + "=" * 80)
print(" GNN MODELS (With Graph)")
print("=" * 80)

# GCN
print("\n[6/6a] Training GCN...")
start = time.time()
gcn_model = GCN(
    in_channels=data.x.shape[1],
    hidden_channels=64,
    out_channels=2,
    num_layers=2,
    dropout=0.3
)
gcn_trainer = GCNTrainer(gcn_model, data, device, lr=0.01, weight_decay=0.0001)
gcn_history = gcn_trainer.fit(epochs=50, patience=10, eval_metric='pr_auc', verbose=False)
_, _, gcn_probs_test = gcn_trainer.evaluate(data.test_mask)
gcn_probs = gcn_probs_test[:, 1].cpu().numpy()
gcn_metrics = compute_metrics(y_test, gcn_probs)
print(f"   Time: {time.time()-start:.1f}s | PR-AUC: {gcn_metrics['pr_auc']:.4f} | ROC-AUC: {gcn_metrics['roc_auc']:.4f}")
results.append(('GCN', 'GNN', gcn_metrics))

# GraphSAGE
print("\n[6/6b] Training GraphSAGE...")
start = time.time()
sage_model = GraphSAGE(
    in_channels=data.x.shape[1],
    hidden_channels=64,
    out_channels=2,
    num_layers=2,
    dropout=0.3
)
sage_trainer = GraphSAGETrainer(sage_model, data, device, lr=0.01, weight_decay=0.0001)
sage_history = sage_trainer.fit(epochs=50, patience=10, eval_metric='pr_auc', verbose=False)
_, _, sage_probs_test = sage_trainer.evaluate(data.test_mask)
sage_probs = sage_probs_test[:, 1].cpu().numpy()
sage_metrics = compute_metrics(y_test, sage_probs)
print(f"   Time: {time.time()-start:.1f}s | PR-AUC: {sage_metrics['pr_auc']:.4f} | ROC-AUC: {sage_metrics['roc_auc']:.4f}")
results.append(('GraphSAGE', 'GNN', sage_metrics))

# GAT
print("\n[6/6c] Training GAT...")
start = time.time()
gat_model = GAT(
    in_channels=data.x.shape[1],
    hidden_channels=64,
    out_channels=2,
    num_layers=2,
    heads=4,
    dropout=0.3
)
gat_trainer = GATTrainer(gat_model, data, device, lr=0.01, weight_decay=0.0001)
gat_history = gat_trainer.fit(epochs=50, patience=10, eval_metric='pr_auc', verbose=False)
_, _, gat_probs_test = gat_trainer.evaluate(data.test_mask)
gat_probs = gat_probs_test[:, 1].cpu().numpy()
gat_metrics = compute_metrics(y_test, gat_probs)
print(f"   Time: {time.time()-start:.1f}s | PR-AUC: {gat_metrics['pr_auc']:.4f} | ROC-AUC: {gat_metrics['roc_auc']:.4f}")
results.append(('GAT', 'GNN', gat_metrics))

# ========================================
# RESULTS SUMMARY
# ========================================

print("\n" + "=" * 80)
print(" FINAL RESULTS - CORRECTED DATASET")
print("=" * 80)

# Sort by PR-AUC
results.sort(key=lambda x: x[2]['pr_auc'], reverse=True)

print(f"\n{'Model':<20} {'Type':<10} {'PR-AUC':<10} {'ROC-AUC':<10} {'F1 Score':<10}")
print("-" * 80)
for model_name, model_type, metrics in results:
    print(f"{model_name:<20} {model_type:<10} {metrics['pr_auc']:<10.4f} {metrics['roc_auc']:<10.4f} {metrics['f1']:<10.4f}")

# Save to CSV
print("\n[*] Saving results to metrics_summary.csv...")
summary_path = Path('reports/metrics_summary.csv')
for model_name, model_type, metrics in results:
    append_metrics_to_csv(
        metrics=metrics,
        filepath=summary_path,
        experiment_name='elliptic-corrected-dataset',
        model_name=model_name,
        split='test'
    )

# Analysis
best_model, best_type, best_metrics = results[0]
print(f"\n🏆 BEST MODEL: {best_model} ({best_type})")
print(f"   PR-AUC: {best_metrics['pr_auc']:.4f}")
print(f"   ROC-AUC: {best_metrics['roc_auc']:.4f}")

# Answer the key question
gnn_best = max([r for r in results if r[1] == 'GNN'], key=lambda x: x[2]['pr_auc'])
tabular_best = max([r for r in results if r[1] == 'Tabular'], key=lambda x: x[2]['pr_auc'])

print(f"\n📊 DOES GRAPH HELP?")
print(f"   Best GNN: {gnn_best[0]} - PR-AUC {gnn_best[2]['pr_auc']:.4f}")
print(f"   Best Tabular: {tabular_best[0]} - PR-AUC {tabular_best[2]['pr_auc']:.4f}")

if gnn_best[2]['pr_auc'] > tabular_best[2]['pr_auc']:
    improvement = (gnn_best[2]['pr_auc'] - tabular_best[2]['pr_auc']) / tabular_best[2]['pr_auc'] * 100
    print(f"   ✅ YES! GNNs are {improvement:.1f}% better (graph structure helps!)")
else:
    gap = (tabular_best[2]['pr_auc'] - gnn_best[2]['pr_auc']) / tabular_best[2]['pr_auc'] * 100
    print(f"   ❌ NO. Tabular models are {gap:.1f}% better (features alone sufficient)")

print("\n" + "=" * 80)
print(" RETRAINING COMPLETE!")
print("=" * 80)
