"""Unit tests for Elliptic data loader and splits."""
import pytest
import numpy as np
import torch
from pathlib import Path

from src.data.elliptic_loader import EllipticDataset
from src.data.splits import (
    create_temporal_splits,
    filter_edges_by_split,
    validate_no_future_leakage
)


class TestEllipticLoader:
    """Tests for EllipticDataset loader."""
    
    @pytest.fixture
    def dataset(self):
        """Load dataset once for all tests."""
        return EllipticDataset(root="data/elliptic")
    
    def test_files_exist(self, dataset):
        """Test that all required files exist."""
        assert dataset.features_path.exists()
        assert dataset.classes_path.exists()
        assert dataset.edges_path.exists()
    
    def test_load_dataset(self, dataset):
        """Test dataset loading."""
        data = dataset.load(verbose=False)
        
        # Check data structure
        assert hasattr(data, 'x')
        assert hasattr(data, 'edge_index')
        assert hasattr(data, 'y')
        assert hasattr(data, 'train_mask')
        assert hasattr(data, 'val_mask')
        assert hasattr(data, 'test_mask')
        assert hasattr(data, 'timestamps')
    
    def test_data_shapes(self, dataset):
        """Test data tensor shapes."""
        data = dataset.load(verbose=False)
        
        # Features should be [N, F]
        assert data.x.dim() == 2
        assert data.x.shape[0] > 0
        assert data.x.shape[1] == 182  # Expected feature count
        
        # Edge index should be [2, E]
        assert data.edge_index.shape[0] == 2
        assert data.edge_index.shape[1] > 0
        
        # Labels should be [N]
        assert data.y.shape[0] == data.x.shape[0]
        
        # Masks should be [N] boolean
        assert data.train_mask.shape[0] == data.x.shape[0]
        assert data.val_mask.shape[0] == data.x.shape[0]
        assert data.test_mask.shape[0] == data.x.shape[0]
    
    def test_label_values(self, dataset):
        """Test label encoding."""
        data = dataset.load(verbose=False)
        
        # Labels should be 0 (legit), 1 (fraud), or -1 (unlabeled)
        unique_labels = torch.unique(data.y)
        assert all(label in [-1, 0, 1] for label in unique_labels)
        
        # Only labeled nodes should be in masks
        labeled = data.y >= 0
        assert (data.train_mask & ~labeled).sum() == 0
        assert (data.val_mask & ~labeled).sum() == 0
        assert (data.test_mask & ~labeled).sum() == 0
    
    def test_split_coverage(self, dataset):
        """Test that splits cover all labeled data without overlap."""
        data = dataset.load(verbose=False)
        
        # No overlap between splits
        assert (data.train_mask & data.val_mask).sum() == 0
        assert (data.train_mask & data.test_mask).sum() == 0
        assert (data.val_mask & data.test_mask).sum() == 0
        
        # All labeled nodes are in exactly one split
        labeled = data.y >= 0
        in_any_split = data.train_mask | data.val_mask | data.test_mask
        assert (labeled == in_any_split).all()
    
    def test_temporal_ordering(self, dataset):
        """Test temporal split ordering."""
        data = dataset.load(verbose=False)
        
        train_times = data.timestamps[data.train_mask]
        val_times = data.timestamps[data.val_mask]
        test_times = data.timestamps[data.test_mask]
        
        # Train should be before val, val before test
        if len(train_times) > 0 and len(val_times) > 0:
            assert train_times.max() < val_times.min()
        
        if len(val_times) > 0 and len(test_times) > 0:
            assert val_times.max() < test_times.min()
    
    def test_splits_json_saved(self, dataset):
        """Test that splits.json is created."""
        dataset.load(verbose=False)
        assert dataset.splits_path.exists()
        
        # Check JSON structure
        import json
        with open(dataset.splits_path) as f:
            splits = json.load(f)
        
        required_keys = [
            'train_time_end', 'val_time_end', 'train_nodes',
            'val_nodes', 'test_nodes', 'total_nodes',
            'total_edges', 'num_features'
        ]
        for key in required_keys:
            assert key in splits


class TestSplitUtilities:
    """Tests for split utility functions."""
    
    def test_create_temporal_splits(self):
        """Test temporal split creation."""
        timestamps = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
        splits = create_temporal_splits(
            timestamps, 
            train_frac=0.6, 
            val_frac=0.2, 
            test_frac=0.2
        )
        
        assert 'train' in splits
        assert 'val' in splits
        assert 'test' in splits
        
        # Check coverage
        assert splits['train'].sum() + splits['val'].sum() + splits['test'].sum() == len(timestamps)
        
        # Check no overlap
        assert (splits['train'] & splits['val']).sum() == 0
        assert (splits['train'] & splits['test']).sum() == 0
        assert (splits['val'] & splits['test']).sum() == 0
    
    def test_filter_edges_by_split(self):
        """Test edge filtering."""
        edge_index = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
        node_mask = np.array([True, True, True, False, False])
        
        filtered = filter_edges_by_split(edge_index, node_mask)
        
        # Only edges with both endpoints in mask should remain
        assert filtered.shape[1] <= edge_index.shape[1]
        assert all(node_mask[filtered[0]])
        assert all(node_mask[filtered[1]])
    
    def test_validate_no_future_leakage(self):
        """Test temporal validation."""
        edge_index = np.array([[0, 1, 2], [1, 2, 3]])
        timestamps = np.array([1, 2, 3, 4])
        
        # Valid case: edges flow forward in time
        assert validate_no_future_leakage(edge_index, timestamps, "test")
        
        # Invalid case: edge points to past
        bad_edge_index = np.array([[2, 1], [0, 0]])
        assert not validate_no_future_leakage(bad_edge_index, timestamps, "test")


class TestNoDataLeakage:
    """Tests specifically for data leakage prevention."""
    
    @pytest.fixture
    def data(self):
        """Load dataset for leakage tests."""
        dataset = EllipticDataset(root="data/elliptic")
        return dataset.load(verbose=False)
    
    def test_no_cross_time_edges_in_train(self, data):
        """Ensure train edges don't cross temporal boundaries."""
        edge_index_np = data.edge_index.numpy()
        train_mask_np = data.train_mask.numpy()
        timestamps_np = data.timestamps.numpy()
        
        # Get train edges
        train_edges = filter_edges_by_split(edge_index_np, train_mask_np)
        
        # Both endpoints should be in train time period
        src_times = timestamps_np[train_edges[0]]
        dst_times = timestamps_np[train_edges[1]]
        
        train_time_end = timestamps_np[train_mask_np].max()
        
        assert (src_times <= train_time_end).all()
        assert (dst_times <= train_time_end).all()
    
    def test_no_future_information_in_val(self, data):
        """Ensure validation doesn't use future data."""
        edge_index_np = data.edge_index.numpy()
        val_mask_np = data.val_mask.numpy()
        timestamps_np = data.timestamps.numpy()
        
        # Get val edges
        val_edges = filter_edges_by_split(edge_index_np, val_mask_np)
        
        # Both endpoints should be in val time period
        src_times = timestamps_np[val_edges[0]]
        dst_times = timestamps_np[val_edges[1]]
        
        train_time_end = timestamps_np[data.train_mask.numpy()].max()
        val_time_end = timestamps_np[val_mask_np].max()
        
        assert (src_times > train_time_end).all()
        assert (dst_times > train_time_end).all()
        assert (src_times <= val_time_end).all()
        assert (dst_times <= val_time_end).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
