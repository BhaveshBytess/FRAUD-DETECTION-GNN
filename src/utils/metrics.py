"""Metrics computation utilities."""
import numpy as np
from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    f1_score,
    roc_curve
)


def compute_metrics(y_true, y_pred_proba, threshold=0.5):
    """
    Compute comprehensive metrics for binary classification.
    
    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred_proba: Predicted probabilities for class 1
        threshold: Classification threshold
        
    Returns:
        dict: Dictionary containing metrics
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    metrics = {
        'pr_auc': average_precision_score(y_true, y_pred_proba),
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
        'f1': f1_score(y_true, y_pred),
        'threshold': threshold
    }
    
    return metrics


def find_best_f1_threshold(y_true, y_pred_proba):
    """
    Find threshold that maximizes F1 score.
    
    Args:
        y_true: Ground truth labels
        y_pred_proba: Predicted probabilities
        
    Returns:
        tuple: (best_threshold, best_f1)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    # Compute F1 for each threshold
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
    
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    return best_threshold, best_f1


def compute_recall_at_k(y_true, y_pred_proba, k_fracs=[0.005, 0.01, 0.02]):
    """
    Compute Recall@K for different values of K.
    
    Args:
        y_true: Ground truth labels
        y_pred_proba: Predicted probabilities
        k_fracs: List of fractions for top-K selection
        
    Returns:
        dict: Recall values for each K
    """
    recalls = {}
    
    # Sort by predicted probability (descending)
    sorted_indices = np.argsort(y_pred_proba)[::-1]
    sorted_labels = y_true[sorted_indices]
    
    total_positives = y_true.sum()
    
    for k_frac in k_fracs:
        k = max(1, int(len(y_true) * k_frac))
        top_k_positives = sorted_labels[:k].sum()
        recall = top_k_positives / total_positives if total_positives > 0 else 0
        recalls[f'recall@{k_frac*100:.1f}%'] = recall
    
    return recalls
