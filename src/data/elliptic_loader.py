"""
Elliptic++ Dataset Loader

Loads Bitcoin transaction graph with temporal splits.
Classes: 1=fraud, 2=legitimate, 3=unlabeled
"""
import os
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from .splits import create_temporal_splits, filter_edges_by_split, validate_no_future_leakage


class EllipticDataset:
    """Elliptic++ Bitcoin transaction graph dataset."""
    
    def __init__(
        self,
        root: str = "data/elliptic",
        train_frac: float = 0.6,
        val_frac: float = 0.2,
        test_frac: float = 0.2
    ):
        """
        Initialize Elliptic dataset loader.
        
        Args:
            root: Root directory containing dataset files
            train_frac: Fraction of timesteps for training
            val_frac: Fraction of timesteps for validation
            test_frac: Fraction of timesteps for testing
        """
        self.root = Path(root)
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac
        
        # File paths
        self.features_path = self.root / "txs_features.csv"
        self.classes_path = self.root / "txs_classes.csv"
        self.edges_path = self.root / "txs_edgelist.csv"
        self.splits_path = self.root / "splits.json"
        
        # Verify files exist
        self._verify_files()
        
        # Load data
        self.data = None
        self.splits_info = None
        
    def _verify_files(self):
        """Verify all required files exist."""
        required_files = [self.features_path, self.classes_path, self.edges_path]
        missing = [f for f in required_files if not f.exists()]
        
        if missing:
            raise FileNotFoundError(
                f"Missing required files in {self.root}:\n" + 
                "\n".join(f"  - {f.name}" for f in missing)
            )
    
    def load(self, verbose: bool = True) -> Data:
        """
        Load and process the dataset.
        
        Args:
            verbose: Print loading progress
            
        Returns:
            PyTorch Geometric Data object with temporal splits
        """
        if verbose:
            print("=" * 60)
            print("Loading Elliptic++ Dataset")
            print("=" * 60)
        
        # Load features
        if verbose:
            print(f"\n[*] Loading features from {self.features_path.name}...")
        features_df = pd.read_csv(self.features_path)
        
        # Load classes
        if verbose:
            print(f"[*] Loading classes from {self.classes_path.name}...")
        classes_df = pd.read_csv(self.classes_path)
        
        # Load edges
        if verbose:
            print(f"[*] Loading edges from {self.edges_path.name}...")
        edges_df = pd.read_csv(self.edges_path)
        
        # Merge features and classes
        if verbose:
            print(f"\n[*] Merging features and labels...")
        data_df = features_df.merge(classes_df, on='txId', how='left')

        # Normalize timestamp column name to 'timestamp'
        ts_candidates = ['Time step','time_step','timestamp','time','timestep']
        for c in ts_candidates:
            if c in data_df.columns:
                if c != 'timestamp':
                    data_df.rename(columns={c: 'timestamp'}, inplace=True)
                break
        else:
            raise KeyError(f"No timestamp column found. Expected one of {ts_candidates}. Columns: {list(data_df.columns)[:25]}")

        # Fill unlabeled as class 3 (will be filtered later)
        data_df['class'] = data_df['class'].fillna(3).astype(int)

        if verbose:
            print(f"   Total transactions: {len(data_df):,}")
            print(f"   Fraud (class=1): {(data_df['class'] == 1).sum():,}")
            print(f"   Legitimate (class=2): {(data_df['class'] == 2).sum():,}")
            print(f"   Unlabeled (class=3): {(data_df['class'] == 3).sum():,}")
        
        # Create tx_id to index mapping
        tx_ids = data_df['txId'].values
        tx_id_to_idx = {tx_id: idx for idx, tx_id in enumerate(tx_ids)}
        
        # Extract features (exclude identifier, timestamp, class)
        feature_cols = [col for col in data_df.columns 
                       if col not in ['txId', 'timestamp', 'class']]
        x = torch.FloatTensor(data_df[feature_cols].values)
        
        # Normalize features (important for GNN stability)
        x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-8)
        
        # Extract unified timestamps
        timestamps = data_df['timestamp'].values
        
        # Convert classes: 1=fraud (1), 2=legit (0), 3=unlabeled (-1)
        y_raw = data_df['class'].values
        y = np.where(y_raw == 1, 1, np.where(y_raw == 2, 0, -1))
        y = torch.LongTensor(y)
        
        # Build edge index
        if verbose:
            print(f"\n[*] Building edge index...")
        
        # Filter edges to known nodes
        valid_edges = edges_df[
            edges_df['txId1'].isin(tx_id_to_idx) & 
            edges_df['txId2'].isin(tx_id_to_idx)
        ]
        
        edge_src = valid_edges['txId1'].map(tx_id_to_idx).values
        edge_dst = valid_edges['txId2'].map(tx_id_to_idx).values
        edge_index = torch.LongTensor(np.vstack([edge_src, edge_dst]))
        
        if verbose:
            print(f"   Total edges: {edge_index.shape[1]:,}")
        
        # Create temporal splits
        if verbose:
            print(f"\n[*] Creating temporal splits...")
        
        splits = create_temporal_splits(
            timestamps, 
            self.train_frac, 
            self.val_frac, 
            self.test_frac
        )
        
        # Create masks for labeled nodes only
        labeled_mask = y >= 0
        
        train_mask = torch.BoolTensor(splits['train'] & labeled_mask.numpy())
        val_mask = torch.BoolTensor(splits['val'] & labeled_mask.numpy())
        test_mask = torch.BoolTensor(splits['test'] & labeled_mask.numpy())
        
        if verbose:
            print(f"   Train: {train_mask.sum():,} labeled nodes (time <= {splits['train_time_end']})")
            print(f"   Val:   {val_mask.sum():,} labeled nodes (time <= {splits['val_time_end']})")
            print(f"   Test:  {test_mask.sum():,} labeled nodes")
            
            # Class balance per split
            for split_name, mask in [('Train', train_mask), ('Val', val_mask), ('Test', test_mask)]:
                if mask.sum() > 0:
                    fraud = (y[mask] == 1).sum().item()
                    legit = (y[mask] == 0).sum().item()
                    total = mask.sum().item()
                    print(f"   {split_name} balance: {fraud:,} fraud ({100*fraud/total:.2f}%), "
                          f"{legit:,} legit ({100*legit/total:.2f}%)")
        
        # Store timestamps
        timestamps_tensor = torch.LongTensor(timestamps)
        
        # Create PyG Data object
        self.data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            timestamps=timestamps_tensor
        )
        
        # Save splits info
        self.splits_info = {
            'train_time_end': int(splits['train_time_end']),
            'val_time_end': int(splits['val_time_end']),
            'train_nodes': int(train_mask.sum()),
            'val_nodes': int(val_mask.sum()),
            'test_nodes': int(test_mask.sum()),
            'total_nodes': len(x),
            'total_edges': int(edge_index.shape[1]),
            'num_features': x.shape[1],
            'train_fraud': int((y[train_mask] == 1).sum()),
            'val_fraud': int((y[val_mask] == 1).sum()),
            'test_fraud': int((y[test_mask] == 1).sum()),
        }
        
        # Save splits to JSON
        self._save_splits()
        
        if verbose:
            print(f"\n[OK] Dataset loaded successfully!")
            print(f"   Features: {x.shape[1]}")
            print(f"   Nodes: {len(x):,}")
            print(f"   Edges: {edge_index.shape[1]:,}")
            print("=" * 60)
        
        return self.data
    
    def _save_splits(self):
        """Save split information to JSON."""
        with open(self.splits_path, 'w') as f:
            json.dump(self.splits_info, f, indent=2)
    
    def get_stats(self) -> Dict:
        """Get dataset statistics."""
        if self.data is None:
            raise RuntimeError("Dataset not loaded. Call load() first.")
        return self.splits_info


def main():
    """CLI entry point for dataset validation."""
    parser = argparse.ArgumentParser(description="Elliptic++ Dataset Loader")
    parser.add_argument(
        '--check', 
        action='store_true', 
        help='Validate dataset and print statistics'
    )
    parser.add_argument(
        '--root', 
        default='data/elliptic',
        help='Root directory containing dataset files'
    )
    
    args = parser.parse_args()
    
    if args.check:
        # Load and validate dataset
        try:
            dataset = EllipticDataset(root=args.root)
            data = dataset.load(verbose=True)
            
            # Additional validation
            print(f"\n[!] Running validation checks...")
            
            # Check for data leakage (edges crossing time boundaries)
            edge_index_np = data.edge_index.numpy()
            timestamps_np = data.timestamps.numpy()
            
            # Split edges by train/val/test
            train_mask_np = data.train_mask.numpy()
            val_mask_np = data.val_mask.numpy()
            test_mask_np = data.test_mask.numpy()
            
            train_edges = filter_edges_by_split(edge_index_np, train_mask_np)
            val_edges = filter_edges_by_split(edge_index_np, val_mask_np)
            test_edges = filter_edges_by_split(edge_index_np, test_mask_np)
            
            print(f"   Train edges (both endpoints in train): {train_edges.shape[1]:,}")
            print(f"   Val edges (both endpoints in val): {val_edges.shape[1]:,}")
            print(f"   Test edges (both endpoints in test): {test_edges.shape[1]:,}")
            
            # Validate temporal consistency
            validate_no_future_leakage(train_edges, timestamps_np, "Train")
            validate_no_future_leakage(val_edges, timestamps_np, "Val")
            validate_no_future_leakage(test_edges, timestamps_np, "Test")
            
            print(f"\n[OK] All validation checks passed!")
            print(f"[*] Splits saved to: {dataset.splits_path}")
            
        except Exception as e:
            print(f"\n[ERROR] Error: {e}")
            raise
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
