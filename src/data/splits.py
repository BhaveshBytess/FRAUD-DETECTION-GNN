"""Temporal split utilities for Elliptic++ dataset."""
import numpy as np
from typing import Dict, Tuple


def create_temporal_splits(
    timestamps: np.ndarray,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    test_frac: float = 0.2
) -> Dict[str, np.ndarray]:
    """
    Create temporal splits based on timestamps.
    
    Args:
        timestamps: Array of timestamps for each node
        train_frac: Fraction for training set
        val_frac: Fraction for validation set
        test_frac: Fraction for test set
        
    Returns:
        Dictionary with 'train', 'val', 'test' boolean masks
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "Fractions must sum to 1"
    
    # Sort timestamps and find boundaries
    sorted_times = np.sort(np.unique(timestamps))
    n_timesteps = len(sorted_times)
    
    train_end_idx = int(n_timesteps * train_frac)
    val_end_idx = int(n_timesteps * (train_frac + val_frac))
    
    train_time_end = sorted_times[train_end_idx - 1] if train_end_idx > 0 else sorted_times[0]
    val_time_end = sorted_times[val_end_idx - 1] if val_end_idx < n_timesteps else sorted_times[-1]
    
    # Create masks
    train_mask = timestamps <= train_time_end
    val_mask = (timestamps > train_time_end) & (timestamps <= val_time_end)
    test_mask = timestamps > val_time_end
    
    splits = {
        'train': train_mask,
        'val': val_mask,
        'test': test_mask,
        'train_time_end': int(train_time_end),
        'val_time_end': int(val_time_end)
    }
    
    return splits


def filter_edges_by_split(
    edge_index: np.ndarray,
    node_mask: np.ndarray
) -> np.ndarray:
    """
    Filter edges so both endpoints are in the split.
    
    Args:
        edge_index: [2, E] array of edge indices
        node_mask: Boolean mask for nodes in this split
        
    Returns:
        Filtered edge_index [2, E']
    """
    # Both source and target must be in the split
    valid_edges = node_mask[edge_index[0]] & node_mask[edge_index[1]]
    return edge_index[:, valid_edges]


def validate_no_future_leakage(
    edge_index: np.ndarray,
    timestamps: np.ndarray,
    split_name: str
) -> bool:
    """
    Verify no edges point from past to future.
    
    Args:
        edge_index: [2, E] edge indices
        timestamps: Node timestamps
        split_name: Name of split for logging
        
    Returns:
        True if valid, False if leakage detected
    """
    src_times = timestamps[edge_index[0]]
    dst_times = timestamps[edge_index[1]]
    
    # All edges should flow forward or same time
    future_leaks = (dst_times < src_times).sum()
    
    if future_leaks > 0:
        print(f"[!] WARNING: {split_name} has {future_leaks} edges pointing to past!")
        return False
    
    return True
