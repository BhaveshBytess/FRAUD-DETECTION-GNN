"""Graph Convolutional Network (GCN) for fraud detection."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    """
    Graph Convolutional Network for binary node classification.
    
    Architecture:
        Input -> GCNConv -> ReLU -> Dropout -> 
        GCNConv -> ReLU -> Dropout -> 
        ... (num_layers) ... ->
        GCNConv -> Logits [N, 2]
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 2,
        num_layers: int = 2,
        dropout: float = 0.4
    ):
        """
        Initialize GCN model.
        
        Args:
            in_channels: Number of input features
            hidden_channels: Hidden dimension size
            out_channels: Number of output classes (2 for binary)
            num_layers: Number of GCN layers
            dropout: Dropout probability
        """
        super(GCN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Build GCN layers
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(in_channels, hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        # Output layer
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_channels, out_channels))
        else:
            # Single layer case
            self.convs[0] = GCNConv(in_channels, out_channels)
    
    def forward(self, x, edge_index):
        """
        Forward pass.
        
        Args:
            x: Node features [N, F]
            edge_index: Edge indices [2, E]
            
        Returns:
            Logits [N, out_channels]
        """
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer (no activation, no dropout)
        x = self.convs[-1](x, edge_index)
        
        return x
    
    def reset_parameters(self):
        """Reset all learnable parameters."""
        for conv in self.convs:
            conv.reset_parameters()


class GCNTrainer:
    """Trainer for GCN model with early stopping."""
    
    def __init__(
        self,
        model: nn.Module,
        data,
        device: str = 'cpu',
        lr: float = 0.001,
        weight_decay: float = 0.0005
    ):
        """
        Initialize trainer.
        
        Args:
            model: GCN model
            data: PyG Data object
            device: Device to train on
            lr: Learning rate
            weight_decay: L2 regularization
        """
        self.model = model.to(device)
        self.data = data.to(device)
        self.device = device
        
        # Compute class weights for imbalanced data
        train_labels = data.y[data.train_mask]
        n_pos = (train_labels == 1).sum().item()
        n_neg = (train_labels == 0).sum().item()
        pos_weight = torch.tensor([n_neg / n_pos], device=device)
        
        # Loss function with class weights
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Early stopping
        self.best_val_metric = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        out = self.model(self.data.x, self.data.edge_index)
        
        # Check for NaN
        if torch.isnan(out).any():
            print("[!] WARNING: NaN detected in forward pass")
            return float('inf')
        
        # Compute loss only on training nodes
        loss = self.criterion(
            out[self.data.train_mask],
            self.data.y[self.data.train_mask]
        )
        
        # Backward pass
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def evaluate(self, mask):
        """
        Evaluate model on a data split.
        
        Args:
            mask: Boolean mask for evaluation split
            
        Returns:
            Tuple of (loss, predictions, probabilities)
        """
        self.model.eval()
        
        # Forward pass
        out = self.model(self.data.x, self.data.edge_index)
        
        # Compute loss
        loss = self.criterion(
            out[mask],
            self.data.y[mask]
        ).item()
        
        # Get predictions and probabilities
        probs = F.softmax(out[mask], dim=1)
        preds = out[mask].argmax(dim=1)
        
        return loss, preds, probs
    
    def fit(
        self,
        epochs: int = 100,
        patience: int = 15,
        eval_metric: str = 'pr_auc',
        verbose: bool = True
    ):
        """
        Train model with early stopping.
        
        Args:
            epochs: Maximum number of epochs
            patience: Early stopping patience
            eval_metric: Metric for early stopping ('pr_auc' or 'roc_auc')
            verbose: Print training progress
            
        Returns:
            Dictionary of training history
        """
        from sklearn.metrics import average_precision_score, roc_auc_score
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_metric': []
        }
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch()
            
            # Evaluate on validation
            val_loss, val_preds, val_probs = self.evaluate(self.data.val_mask)
            
            # Compute validation metric
            val_labels = self.data.y[self.data.val_mask].cpu().numpy()
            val_probs_np = val_probs[:, 1].cpu().numpy()
            
            # Check for NaN in predictions
            if np.isnan(val_probs_np).any():
                print(f"[!] WARNING: NaN in predictions at epoch {epoch+1}, skipping")
                continue
            
            if eval_metric == 'pr_auc':
                val_metric = average_precision_score(val_labels, val_probs_np)
            else:
                val_metric = roc_auc_score(val_labels, val_probs_np)
            
            # Record history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_metric'].append(val_metric)
            
            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:03d}: "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, "
                      f"Val {eval_metric.upper()}: {val_metric:.4f}")
            
            # Early stopping check
            if val_metric > self.best_val_metric:
                self.best_val_metric = val_metric
                self.best_epoch = epoch
                self.patience_counter = 0
                # Save best model state
                self.best_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= patience:
                if verbose:
                    print(f"\n[*] Early stopping at epoch {epoch+1}")
                    print(f"   Best {eval_metric.upper()}: {self.best_val_metric:.4f} at epoch {self.best_epoch+1}")
                break
        
        # Restore best model
        self.model.load_state_dict(self.best_state)
        
        return history
