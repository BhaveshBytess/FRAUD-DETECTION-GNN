"""
GAT (Graph Attention Network) model for fraud detection.

GAT uses attention mechanisms to learn adaptive importance weights for
different neighbors, making it robust to noisy graph structures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_recall_curve
)


class GAT(nn.Module):
    """
    Graph Attention Network with multi-head attention.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 2,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.4,
        concat_heads: bool = True
    ):
        """
        Initialize GAT model.
        
        Args:
            in_channels: Number of input features
            hidden_channels: Hidden dimension size (per head)
            out_channels: Number of output classes (2 for binary)
            num_layers: Number of GAT layers
            heads: Number of attention heads
            dropout: Dropout probability
            concat_heads: If True, concatenate heads; else average
        """
        super(GAT, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.concat_heads = concat_heads
        
        # Build GAT layers
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(
            GATConv(
                in_channels,
                hidden_channels,
                heads=heads,
                dropout=dropout,
                concat=concat_heads
            )
        )
        
        # Hidden layers
        input_dim = hidden_channels * heads if concat_heads else hidden_channels
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    input_dim,
                    hidden_channels,
                    heads=heads,
                    dropout=dropout,
                    concat=concat_heads
                )
            )
        
        # Output layer (average heads for output)
        if num_layers > 1:
            self.convs.append(
                GATConv(
                    input_dim,
                    out_channels,
                    heads=heads,
                    dropout=dropout,
                    concat=False  # Average heads for output
                )
            )
        else:
            # Single layer case
            self.convs[0] = GATConv(
                in_channels,
                out_channels,
                heads=heads,
                dropout=dropout,
                concat=False
            )
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weights."""
        for conv in self.convs:
            conv.reset_parameters()
    
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
            x = F.elu(x)  # ELU works better with GAT
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer (no activation, no dropout)
        x = self.convs[-1](x, edge_index)
        
        return x
    
    def get_num_params(self):
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class GATTrainer:
    """
    Trainer for GAT model with early stopping.
    """
    
    def __init__(
        self,
        model: GAT,
        data,
        device: str = 'cpu',
        lr: float = 0.005,
        weight_decay: float = 0.0005
    ):
        """
        Initialize trainer.
        
        Args:
            model: GAT model
            data: PyG Data object with x, edge_index, y, masks
            device: Device to train on ('cpu' or 'cuda')
            lr: Learning rate (GAT often needs higher LR)
            weight_decay: L2 regularization
        """
        self.model = model
        self.data = data
        self.device = device
        
        # Move model and data to device
        self.model = self.model.to(device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Early stopping
        self.best_val_metric = 0
        self.best_epoch = 0
        self.best_state = None
    
    def train_epoch(self, train_mask):
        """Train for one epoch."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        out = self.model(self.data.x, self.data.edge_index)
        loss = self.criterion(out[train_mask], self.data.y[train_mask])
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def evaluate(self, mask):
        """Evaluate model on given mask."""
        self.model.eval()
        
        out = self.model(self.data.x, self.data.edge_index)
        
        # Check for NaN
        if torch.isnan(out).any():
            return None, None, None
        
        # Get predictions
        probs = F.softmax(out[mask], dim=1)[:, 1].cpu().numpy()
        labels = self.data.y[mask].cpu().numpy()
        
        # Check for NaN in probs
        if np.isnan(probs).any():
            return None, None, None
        
        # Compute metrics
        loss = self.criterion(out[mask], self.data.y[mask]).item()
        pr_auc = average_precision_score(labels, probs)
        roc_auc = roc_auc_score(labels, probs)
        
        return loss, pr_auc, roc_auc
    
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
            eval_metric: Metric to use for early stopping ('pr_auc' or 'roc_auc')
            verbose: Print progress
            
        Returns:
            Dictionary with training history
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_pr_auc': [],
            'val_roc_auc': []
        }
        
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(self.data.train_mask)
            
            # Validate
            val_loss, val_pr_auc, val_roc_auc = self.evaluate(self.data.val_mask)
            
            # Skip if NaN
            if val_pr_auc is None:
                if verbose:
                    print(f"Epoch {epoch+1:03d}: NaN detected, skipping...")
                continue
            
            # Record history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_pr_auc'].append(val_pr_auc)
            history['val_roc_auc'].append(val_roc_auc)
            
            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:03d}: Train Loss={train_loss:.4f}, "
                      f"Val Loss={val_loss:.4f}, Val PR-AUC={val_pr_auc:.4f}, "
                      f"Val ROC-AUC={val_roc_auc:.4f}")
            
            # Early stopping
            current_metric = val_pr_auc if eval_metric == 'pr_auc' else val_roc_auc
            
            if current_metric > self.best_val_metric:
                self.best_val_metric = current_metric
                self.best_epoch = epoch
                patience_counter = 0
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    print(f"Best {eval_metric}: {self.best_val_metric:.4f} at epoch {self.best_epoch+1}")
                break
        
        # Load best model
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        
        return history
    
    @torch.no_grad()
    def test(self, test_mask):
        """
        Evaluate on test set.
        
        Returns:
            Dictionary with test metrics
        """
        self.model.eval()
        
        out = self.model(self.data.x, self.data.edge_index)
        probs = F.softmax(out[test_mask], dim=1)[:, 1].cpu().numpy()
        labels = self.data.y[test_mask].cpu().numpy()
        
        # Compute metrics
        pr_auc = average_precision_score(labels, probs)
        roc_auc = roc_auc_score(labels, probs)
        
        # Find best threshold on validation
        val_out = self.model(self.data.x, self.data.edge_index)
        val_probs = F.softmax(val_out[self.data.val_mask], dim=1)[:, 1].cpu().numpy()
        val_labels = self.data.y[self.data.val_mask].cpu().numpy()
        
        precision, recall, thresholds = precision_recall_curve(val_labels, val_probs)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_threshold = thresholds[np.argmax(f1_scores)] if len(thresholds) > 0 else 0.5
        
        # Apply threshold
        preds = (probs >= best_threshold).astype(int)
        f1 = f1_score(labels, preds)
        
        # Recall@K
        def recall_at_k(y_true, y_score, k_frac=0.01):
            k = max(1, int(len(y_true) * k_frac))
            top_k_idx = np.argsort(y_score)[-k:]
            return y_true[top_k_idx].sum() / y_true.sum()
        
        recall_05 = recall_at_k(labels, probs, 0.005)
        recall_10 = recall_at_k(labels, probs, 0.01)
        recall_20 = recall_at_k(labels, probs, 0.02)
        
        return {
            'pr_auc': pr_auc,
            'roc_auc': roc_auc,
            'f1': f1,
            'threshold': best_threshold,
            'recall@0.5%': recall_05,
            'recall@1.0%': recall_10,
            'recall@2.0%': recall_20
        }
