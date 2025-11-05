"""Logging utilities."""
import json
import csv
import os
from datetime import datetime
from pathlib import Path


def save_metrics_json(metrics, filepath):
    """
    Save metrics dictionary to JSON file.
    
    Args:
        metrics: Dictionary of metrics
        filepath: Path to save JSON file
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)


def append_metrics_to_csv(metrics, filepath, experiment_name, model_name, split='test'):
    """
    Append metrics to summary CSV file.
    
    Args:
        metrics: Dictionary of metrics
        filepath: Path to CSV file
        experiment_name: Name of experiment
        model_name: Name of model
        split: Data split (train/val/test)
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    # Check if file exists
    file_exists = os.path.isfile(filepath)
    
    row = {
        'timestamp': int(datetime.now().timestamp()),
        'experiment': experiment_name,
        'model': model_name,
        'split': split,
        'pr_auc': metrics.get('pr_auc', 0),
        'roc_auc': metrics.get('roc_auc', 0),
        'f1': metrics.get('f1', 0),
        'recall@1%': metrics.get('recall@1.0%', 0)
    }
    
    with open(filepath, 'a', newline='') as f:
        fieldnames = ['timestamp', 'experiment', 'model', 'split', 'pr_auc', 'roc_auc', 'f1', 'recall@1%']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(row)


def load_config(config_path):
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        dict: Configuration dictionary
    """
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
