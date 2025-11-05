"""Unit tests for GNN model architectures."""
import pytest
import torch
from src.models.gcn import GCN


class TestGCNModel:
    """Tests for GCN model."""
    
    def test_gcn_init(self):
        """Test GCN initialization."""
        model = GCN(
            in_channels=10,
            hidden_channels=32,
            out_channels=2,
            num_layers=2,
            dropout=0.4
        )
        assert model is not None
        assert hasattr(model, 'convs')
        assert len(model.convs) == 2
    
    def test_gcn_forward_shape(self):
        """Test GCN forward pass output shape."""
        n_nodes = 100
        n_features = 10
        n_edges = 200
        
        model = GCN(
            in_channels=n_features,
            hidden_channels=32,
            out_channels=2,
            num_layers=2
        )
        
        # Create dummy data
        x = torch.randn(n_nodes, n_features)
        edge_index = torch.randint(0, n_nodes, (2, n_edges))
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            out = model(x, edge_index)
        
        # Check output shape
        assert out.shape == (n_nodes, 2)
    
    def test_gcn_single_layer(self):
        """Test GCN with single layer."""
        model = GCN(
            in_channels=10,
            hidden_channels=32,
            out_channels=2,
            num_layers=1
        )
        
        assert len(model.convs) == 1
        
        # Test forward
        x = torch.randn(50, 10)
        edge_index = torch.randint(0, 50, (2, 100))
        
        model.eval()
        with torch.no_grad():
            out = model(x, edge_index)
        
        assert out.shape == (50, 2)
    
    def test_gcn_multi_layer(self):
        """Test GCN with multiple layers."""
        for num_layers in [2, 3, 4]:
            model = GCN(
                in_channels=10,
                hidden_channels=32,
                out_channels=2,
                num_layers=num_layers
            )
            
            assert len(model.convs) == num_layers
    
    def test_gcn_parameters_count(self):
        """Test that GCN has trainable parameters."""
        model = GCN(
            in_channels=10,
            hidden_channels=32,
            out_channels=2,
            num_layers=2
        )
        
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0
        
        # Check that parameters are trainable
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable_params == n_params
    
    def test_gcn_reset_parameters(self):
        """Test parameter reset."""
        model = GCN(in_channels=10, hidden_channels=32, out_channels=2, num_layers=2)
        
        # Get initial parameters
        initial_params = [p.clone() for p in model.parameters()]
        
        # Reset
        model.reset_parameters()
        
        # Check that parameters changed
        reset_params = list(model.parameters())
        
        # At least some parameters should be different
        different = False
        for init_p, reset_p in zip(initial_params, reset_params):
            if not torch.allclose(init_p, reset_p):
                different = True
                break
        
        assert different, "Parameters should change after reset"
    
    def test_gcn_training_mode(self):
        """Test training vs eval mode."""
        model = GCN(in_channels=10, hidden_channels=32, out_channels=2, num_layers=2, dropout=0.5)
        
        x = torch.randn(50, 10)
        edge_index = torch.randint(0, 50, (2, 100))
        
        # Training mode
        model.train()
        out_train_1 = model(x, edge_index)
        out_train_2 = model(x, edge_index)
        
        # Outputs should be different in training mode due to dropout
        assert not torch.allclose(out_train_1, out_train_2)
        
        # Eval mode
        model.eval()
        with torch.no_grad():
            out_eval_1 = model(x, edge_index)
            out_eval_2 = model(x, edge_index)
        
        # Outputs should be identical in eval mode
        assert torch.allclose(out_eval_1, out_eval_2)
    
    def test_gcn_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = GCN(in_channels=10, hidden_channels=32, out_channels=2, num_layers=2)
        
        x = torch.randn(50, 10, requires_grad=True)
        edge_index = torch.randint(0, 50, (2, 100))
        target = torch.randint(0, 2, (50,))
        
        # Forward pass
        out = model(x, edge_index)
        loss = torch.nn.functional.cross_entropy(out, target)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist
        assert any(p.grad is not None for p in model.parameters())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
