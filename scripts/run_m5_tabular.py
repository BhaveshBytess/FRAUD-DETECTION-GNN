"""
M5 - Tabular Baselines Training Script
Train traditional ML models on node features only (no graph)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    f1_score,
    classification_report
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("M5 - TABULAR BASELINES")
print("="*80)

# Setup paths
data_dir = Path(r'C:\Users\oumme\OneDrive\Desktop\FRAUD-DETECTION-GNN\Elliptic++ Dataset')
reports_dir = Path(r'C:\Users\oumme\OneDrive\Desktop\FRAUD-DETECTION-GNN\reports')
plots_dir = reports_dir / 'plots'
plots_dir.mkdir(parents=True, exist_ok=True)

print(f"\n1. Loading data from: {data_dir}")

# Load features and classes
features_df = pd.read_csv(data_dir / 'txs_features.csv')
classes_df = pd.read_csv(data_dir / 'txs_classes.csv')

print(f"Features shape: {features_df.shape}")
print(f"Classes shape: {classes_df.shape}")

# Merge
df = features_df.merge(classes_df, on='txId', how='left')
print(f"Merged shape: {df.shape}")

# Find timestamp column
time_col = None
for col in ['timestamp', 'Time step', 'time_step']:
    if col in df.columns:
        time_col = col
        break

if time_col is None:
    raise ValueError(f"No timestamp column found! Columns: {df.columns.tolist()[:10]}...")

# Normalize column name
if time_col != 'timestamp':
    df['timestamp'] = df[time_col]

print(f"Timestamp column: '{time_col}' -> 'timestamp'")
print(f"Timestamp range: {df['timestamp'].min()} to {df['timestamp'].max()}")

# Filter to labeled transactions only
df_labeled = df[df['class'].isin([1, 2])].copy()

# Map class: 1=licit (0), 2=illicit (1)
df_labeled['label'] = (df_labeled['class'] == 2).astype(int)

print(f"\nLabeled transactions: {len(df_labeled):,}")
print(f"Fraud percentage: {df_labeled['label'].mean()*100:.2f}%")

print("\n2. Creating temporal splits...")

# Sort by timestamp
df_labeled = df_labeled.sort_values('timestamp')

# Split: 60% train, 20% val, 20% test (temporal)
n = len(df_labeled)
train_size = int(0.6 * n)
val_size = int(0.2 * n)

train_df = df_labeled.iloc[:train_size]
val_df = df_labeled.iloc[train_size:train_size+val_size]
test_df = df_labeled.iloc[train_size+val_size:]

print(f"Train: {len(train_df):,} ({len(train_df)/n*100:.1f}%) | Fraud: {train_df['label'].mean()*100:.2f}%")
print(f"Val:   {len(val_df):,} ({len(val_df)/n*100:.1f}%) | Fraud: {val_df['label'].mean()*100:.2f}%")
print(f"Test:  {len(test_df):,} ({len(test_df)/n*100:.1f}%) | Fraud: {test_df['label'].mean()*100:.2f}%")

print("\n3. Preparing features...")

# Identify feature columns
exclude_cols = ['txId', 'timestamp', 'Time step', 'time_step', 'class', 'label']
feature_cols = [col for col in df_labeled.columns if col not in exclude_cols]

print(f"Number of features: {len(feature_cols)}")

# Extract features and labels
X_train = train_df[feature_cols].values
y_train = train_df['label'].values

X_val = val_df[feature_cols].values
y_val = val_df['label'].values

X_test = test_df[feature_cols].values
y_test = test_df['label'].values

# Handle NaN/inf values
X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"Train: X={X_train_scaled.shape}, y={y_train.shape}")
print(f"Val:   X={X_val_scaled.shape}, y={y_val.shape}")
print(f"Test:  X={X_test_scaled.shape}, y={y_test.shape}")

# Helper functions
def evaluate_model(y_true, y_pred_proba, model_name):
    """Evaluate model with same metrics as GNN models"""
    pr_auc = average_precision_score(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    # F1 score (threshold at 0.5)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    f1 = f1_score(y_true, y_pred)
    
    # Recall@1% (top 1% predictions)
    top_1pct_idx = np.argsort(y_pred_proba)[::-1][:int(len(y_pred_proba)*0.01)]
    recall_at_1pct = y_true[top_1pct_idx].mean()
    
    metrics = {
        'model': model_name,
        'pr_auc': float(pr_auc),
        'roc_auc': float(roc_auc),
        'f1_score': float(f1),
        'recall_at_1pct': float(recall_at_1pct)
    }
    
    print(f"\n{model_name} Results:")
    print(f"  PR-AUC:      {pr_auc:.4f}")
    print(f"  ROC-AUC:     {roc_auc:.4f}")
    print(f"  F1 Score:    {f1:.4f}")
    print(f"  Recall@1%:   {recall_at_1pct:.4f}")
    
    return metrics

def save_metrics(metrics, filename):
    """Save metrics to JSON file"""
    filepath = reports_dir / filename
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved to: {filepath.name}")

# Train models
all_results = []

# 1. Logistic Regression
print("\n" + "="*80)
print("4. Training Logistic Regression...")
print("="*80)

lr_model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

lr_model.fit(X_train_scaled, y_train)
lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
lr_metrics = evaluate_model(y_test, lr_proba, 'Logistic Regression')
save_metrics(lr_metrics, 'logistic_regression_metrics.json')
all_results.append(lr_metrics)

# 2. Random Forest
print("\n" + "="*80)
print("5. Training Random Forest...")
print("="*80)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=4,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    verbose=0
)

rf_model.fit(X_train_scaled, y_train)
rf_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
rf_metrics = evaluate_model(y_test, rf_proba, 'Random Forest')
save_metrics(rf_metrics, 'random_forest_metrics.json')
all_results.append(rf_metrics)

# 3. XGBoost
print("\n" + "="*80)
print("6. Training XGBoost...")
print("="*80)

scale_pos_weight = len(y_train) / y_train.sum() - 1
print(f"Scale pos weight: {scale_pos_weight:.2f}")

xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=10,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    tree_method='hist',
    verbosity=0
)

xgb_model.fit(
    X_train_scaled, y_train,
    eval_set=[(X_val_scaled, y_val)],
    verbose=False
)

xgb_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]
xgb_metrics = evaluate_model(y_test, xgb_proba, 'XGBoost')
save_metrics(xgb_metrics, 'xgboost_metrics.json')
all_results.append(xgb_metrics)

# 4. MLP
print("\n" + "="*80)
print("7. Training MLP...")
print("="*80)

mlp_model = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    batch_size=256,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=100,
    random_state=42,
    verbose=False,
    early_stopping=True,
    validation_fraction=0.1
)

mlp_model.fit(X_train_scaled, y_train)
mlp_proba = mlp_model.predict_proba(X_test_scaled)[:, 1]
mlp_metrics = evaluate_model(y_test, mlp_proba, 'MLP')
save_metrics(mlp_metrics, 'mlp_metrics.json')
all_results.append(mlp_metrics)

# Compare with GNN models
print("\n" + "="*80)
print("8. Comparing with GNN models...")
print("="*80)

gnn_results = [
    {'model': 'GCN', 'pr_auc': 0.1976, 'roc_auc': 0.7627, 'f1_score': 0.2487, 'recall_at_1pct': 0.0613},
    {'model': 'GraphSAGE', 'pr_auc': 0.4483, 'roc_auc': 0.8210, 'f1_score': 0.4527, 'recall_at_1pct': 0.1478},
    {'model': 'GAT', 'pr_auc': 0.1839, 'roc_auc': 0.7942, 'f1_score': 0.2901, 'recall_at_1pct': 0.0126}
]

all_results.extend(gnn_results)

# Create comparison dataframe
df_results = pd.DataFrame(all_results)
df_results = df_results.sort_values('pr_auc', ascending=False)

print("\n" + "="*80)
print("FINAL RESULTS - ALL MODELS")
print("="*80)
print(df_results.to_string(index=False))

# Save to CSV
csv_path = reports_dir / 'all_models_comparison.csv'
df_results.to_csv(csv_path, index=False)
print(f"\nSaved comparison to: {csv_path.name}")

# Create visualization
print("\n9. Creating comparison plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

metrics_to_plot = ['pr_auc', 'roc_auc', 'f1_score', 'recall_at_1pct']
titles = ['PR-AUC (Higher is Better)', 'ROC-AUC (Higher is Better)', 
          'F1 Score (Higher is Better)', 'Recall@1% (Higher is Better)']

for idx, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
    ax = axes[idx // 2, idx % 2]
    
    # Sort by metric for this plot
    df_sorted = df_results.sort_values(metric, ascending=True)
    
    # Color: green for GNN, blue for tabular
    colors = ['green' if model in ['GCN', 'GraphSAGE', 'GAT'] else 'blue' 
              for model in df_sorted['model']]
    
    ax.barh(df_sorted['model'], df_sorted[metric], color=colors, alpha=0.7)
    ax.set_xlabel(metric.upper())
    ax.set_title(title)
    ax.grid(axis='x', alpha=0.3)
    
    # Add values on bars
    for i, (model, value) in enumerate(zip(df_sorted['model'], df_sorted[metric])):
        ax.text(value, i, f' {value:.4f}', va='center', fontsize=9)

plt.tight_layout()
plot_path = plots_dir / 'all_models_comparison.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"Saved plot to: {plot_path.name}")
plt.close()

# Analysis
print("\n" + "="*80)
print("GRAPH STRUCTURE VALUE ANALYSIS")
print("="*80)

tabular_models = df_results[~df_results['model'].isin(['GCN', 'GraphSAGE', 'GAT'])]
best_tabular = tabular_models.loc[tabular_models['pr_auc'].idxmax()]

graphsage = df_results[df_results['model'] == 'GraphSAGE'].iloc[0]

improvement = ((graphsage['pr_auc'] - best_tabular['pr_auc']) / best_tabular['pr_auc']) * 100

print(f"\nBest Tabular Model:  {best_tabular['model']}")
print(f"  PR-AUC: {best_tabular['pr_auc']:.4f}")
print(f"\nBest Graph Model:    GraphSAGE")
print(f"  PR-AUC: {graphsage['pr_auc']:.4f}")
print(f"\nImprovement: {improvement:+.1f}%")

if improvement > 20:
    conclusion = "SIGNIFICANT"
    recommendation = "GNNs are essential for fraud detection on this dataset."
elif improvement > 5:
    conclusion = "MODERATE"
    recommendation = "Consider ensemble of GNN + tabular models."
else:
    conclusion = "MINIMAL"
    recommendation = "Tabular models may be sufficient."

print(f"\n✅ CONCLUSION: Graph structure adds {conclusion} value!")
print(f"   → {recommendation}")

print("\n" + "="*80)
print("M5 - COMPLETE!")
print("="*80)
print("\nDeliverables:")
print("  ✅ 4 tabular models trained and evaluated")
print("  ✅ Metrics saved to reports/*.json")
print("  ✅ Comparison CSV: reports/all_models_comparison.csv")
print("  ✅ Visualization: reports/plots/all_models_comparison.png")
print("  ✅ Graph value analysis complete")
print("\nNext: M6 - Final verification and documentation")
print("="*80)
