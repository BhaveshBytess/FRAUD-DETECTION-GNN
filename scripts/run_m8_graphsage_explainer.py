"""
Runs GNNExplainer on the locally trained GraphSAGE (local-only) checkpoint.
Outputs:
  - JSON summary of top features per explained node
  - Subgraph visualization PNGs
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.nn.models import GNNExplainer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.elliptic_loader import EllipticDataset
from src.data.feature_groups import LOCAL_FEATURES
from src.models.graphsage import GraphSAGE


def select_nodes(logits: torch.Tensor, mask: torch.Tensor, labels: torch.Tensor, k: int = 5):
    probs = torch.softmax(logits, dim=1)[:, 1]
    mask_idx = torch.nonzero(mask, as_tuple=False).view(-1)
    mask_probs = probs[mask_idx]
    sorted_idx = mask_idx[torch.argsort(mask_probs, descending=True)]
    # prefer labeled fraud nodes
    fraud_idx = sorted_idx[labels[sorted_idx] == 1]
    if len(fraud_idx) >= k:
        return fraud_idx[:k].tolist()
    top_nodes = sorted_idx[:k].tolist()
    return top_nodes


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = EllipticDataset(feature_subset=LOCAL_FEATURES)
    data = dataset.load(verbose=False)
    data = data.to(device)

    ckpt_path = PROJECT_ROOT / "checkpoints" / "graphsage_local_only_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError("graphsage_local_only_best.pt not found; run scripts/run_m8_graphsage_local_only.py first.")

    model = GraphSAGE(
        in_channels=data.x.shape[1],
        hidden_channels=128,
        out_channels=2,
        num_layers=2,
        dropout=0.4,
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    with torch.no_grad():
        logits = model(data.x, data.edge_index)

    node_ids = select_nodes(logits.cpu(), data.test_mask.cpu(), data.y.cpu(), k=5)
    print(f"Explaining nodes: {node_ids}")

    explainer = GNNExplainer(model, epochs=100, lr=0.01, return_type="log_probs")
    reports_dir = PROJECT_ROOT / "reports"
    plots_dir = reports_dir / "plots"
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    explanations = []
    for node_id in node_ids:
        node_feat_mask, edge_mask = explainer.explain_node(
            node_id,
            data.x,
            data.edge_index,
        )
        feat_mask = node_feat_mask.cpu().numpy()
        top_feat_idx = np.argsort(feat_mask)[::-1][:10]
        top_features = [
            {"feature": LOCAL_FEATURES[i], "importance": float(feat_mask[i])}
            for i in top_feat_idx
        ]

        fig, ax = explainer.visualize_subgraph(
            node_id,
            data.edge_index.cpu(),
            edge_mask.cpu(),
            y=data.y.cpu(),
        )
        fig_path = plots_dir / f"m8_graphsage_explainer_node{node_id}.png"
        plt.savefig(fig_path, dpi=150)
        plt.close(fig)

        explanations.append(
            {
                "node_id": int(node_id),
                "label": int(data.y[node_id].item()),
                "top_features": top_features,
                "plot_path": str(fig_path),
            }
        )

    out_path = reports_dir / "m8_graphsage_explanations.json"
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(explanations, fp, indent=2)

    print(f"Saved explanation summary to {out_path}")


if __name__ == "__main__":
    main()
