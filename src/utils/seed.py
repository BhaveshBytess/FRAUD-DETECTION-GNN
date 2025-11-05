"""Seed utilities for reproducibility."""
import random
import numpy as np
import torch


def set_all_seeds(seed: int = 42):
    """
    Set seeds for Python, NumPy, and PyTorch for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Note: torch.use_deterministic_algorithms(True) can cause issues with some operations
    # Commenting out for compatibility with PyTorch Geometric on CPU
    # try:
    #     torch.use_deterministic_algorithms(True)
    # except AttributeError:
    #     pass
