"""Train GCN model - Quick script version of notebook."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve

from src.data import EllipticDataset
from src.models.gcn import GCN, GCNTrainer
from src.utils.seed import set_all_seeds
from src.utils.metrics import compute_metrics, find_best_f1_threshold, compute_recall_at_k
from src.utils.logger import save_metrics_json, append_metrics_to_csv

# Set seed for reproducibility
SEED = 42
set_all_seeds(SEED)
print(f"[OK] Seeds set to {SEED}\n")

# Load dataset
print("=" * 60)
print("Loading Elliptic++ Dataset")
print("=" * 60)
dataset = EllipticDataset(root='data/elliptic')
data = dataset.load(verbose=True)

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\n[*] Using device: {device}")

# Model configuration
config = {
    'in_channels': data.x.shape[1],
    'hidden_channels': 128,
    'out_channels': 2,
    'num_layers': 2,
    'dropout': 0.4
}

# Initialize model
model = GCN(**config)
print(f"\n[*] GCN Model initialized")
print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Initialize trainer
trainer = GCNTrainer(
    model=model,
    data=data,
    device=device,
    lr=0.001,
    weight_decay=0.0005
)

# Train
print("\n" + "=" * 60)
print("Training GCN Model")
print("=" * 60)
history = trainer.fit(epochs=100, patience=15, eval_metric='pr_auc', verbose=True)

print(f"\n[OK] Training complete!")
print(f"   Best validation PR-AUC: {trainer.best_val_metric:.4f}")
print(f"   Best epoch: {trainer.best_epoch + 1}")

# Evaluate on test set
print("\n" + "=" * 60)
print("Evaluating on Test Set")
print("=" * 60)

test_loss, test_preds, test_probs = trainer.evaluate(data.test_mask)
test_labels = data.y[data.test_mask].cpu().numpy()
test_probs_fraud = test_probs[:, 1].cpu().numpy()

# Find best threshold on validation
val_loss, val_preds, val_probs = trainer.evaluate(data.val_mask)
val_labels = data.y[data.val_mask].cpu().numpy()
val_probs_fraud = val_probs[:, 1].cpu().numpy()
best_threshold, best_f1_val = find_best_f1_threshold(val_labels, val_probs_fraud)

# Compute metrics
test_metrics = compute_metrics(test_labels, test_probs_fraud, threshold=best_threshold)
recall_at_k = compute_recall_at_k(test_labels, test_probs_fraud, k_fracs=[0.005, 0.01, 0.02])
test_metrics.update(recall_at_k)

print(f"\n[*] Test Set Metrics:")
print(f"   PR-AUC:      {test_metrics['pr_auc']:.4f} (primary)")
print(f"   ROC-AUC:     {test_metrics['roc_auc']:.4f}")
print(f"   F1 Score:    {test_metrics['f1']:.4f}")
print(f"   Threshold:   {test_metrics['threshold']:.4f}")
print(f"\n   Recall@0.5%: {test_metrics['recall@0.5%']:.4f}")
print(f"   Recall@1.0%: {test_metrics['recall@1.0%']:.4f}")
print(f"   Recall@2.0%: {test_metrics['recall@2.0%']:.4f}")

# Save artifacts
print("\n" + "=" * 60)
print("Saving Artifacts")
print("=" * 60)

# Create directories
Path('reports/plots').mkdir(parents=True, exist_ok=True)
Path('checkpoints').mkdir(parents=True, exist_ok=True)

# Save checkpoint
checkpoint_path = Path('checkpoints/gcn_best.pt')
torch.save({
    'model_state_dict': model.state_dict(),
    'config': config,
    'metrics': test_metrics,
    'best_epoch': trainer.best_epoch,
    'seed': SEED
}, checkpoint_path)
print(f"[OK] Model checkpoint: {checkpoint_path}")

# Save metrics JSON
metrics_json_path = Path('reports/gcn_metrics.json')
save_metrics_json(test_metrics, metrics_json_path)
print(f"[OK] Metrics JSON: {metrics_json_path}")

# Append to summary CSV
summary_csv_path = Path('reports/metrics_summary.csv')
append_metrics_to_csv(
    metrics=test_metrics,
    filepath=summary_csv_path,
    experiment_name='elliptic-gnn-baselines',
    model_name='GCN',
    split='test'
)
print(f"[OK] Summary CSV: {summary_csv_path}")

# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history['train_loss'], label='Train', linewidth=2)
axes[0].plot(history['val_loss'], label='Val', linewidth=2)
axes[0].axvline(trainer.best_epoch, color='r', linestyle='--', alpha=0.5)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training History')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(history['val_metric'], color='green', linewidth=2)
axes[1].axvline(trainer.best_epoch, color='r', linestyle='--', alpha=0.5)
axes[1].axhline(trainer.best_val_metric, color='g', linestyle=':', alpha=0.5)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('PR-AUC')
axes[1].set_title('Validation PR-AUC')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('reports/plots/gcn_training_history.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"[OK] Training plot: reports/plots/gcn_training_history.png")

# Plot PR and ROC curves
precision, recall, _ = precision_recall_curve(test_labels, test_probs_fraud)
fpr, tpr, _ = roc_curve(test_labels, test_probs_fraud)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(recall, precision, linewidth=2.5, label=f'GCN (PR-AUC={test_metrics["pr_auc"]:.4f})')
axes[0].axhline(test_labels.mean(), color='red', linestyle='--', alpha=0.7, label='Baseline')
axes[0].set_xlabel('Recall')
axes[0].set_ylabel('Precision')
axes[0].set_title('Precision-Recall Curve', fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(fpr, tpr, linewidth=2.5, label=f'GCN (ROC-AUC={test_metrics["roc_auc"]:.4f})')
axes[1].plot([0, 1], [0, 1], color='red', linestyle='--', alpha=0.7, label='Baseline')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve', fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('reports/plots/gcn_pr_roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"[OK] Curves plot: reports/plots/gcn_pr_roc_curves.png")

# Final summary
print("\n" + "=" * 60)
print("GCN BASELINE - COMPLETE")
print("=" * 60)
print(f"Test PR-AUC: {test_metrics['pr_auc']:.4f}")
print(f"All artifacts saved to reports/ and checkpoints/")
print("=" * 60)
