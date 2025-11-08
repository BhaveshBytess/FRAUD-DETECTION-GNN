"""
Gradient-based feature saliency for GraphSAGE (local-only) to support M8.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.elliptic_loader import EllipticDataset
from src.data.feature_groups import LOCAL_FEATURES
from src.models.graphsage import GraphSAGE


def select_top_nodes(logits, mask, k=5):
    probs = torch.softmax(logits, dim=1)[:, 1]
    idx = torch.nonzero(mask, as_tuple=False).view(-1)
    subset = idx[torch.argsort(probs[idx], descending=True)]
    return subset[:k].tolist()


def neighbor_summary(edge_index: torch.Tensor, node_id: int, probs: torch.Tensor, top_k: int = 10):
    src, dst = edge_index
    mask = (src == node_id) | (dst == node_id)
    neighbors = torch.unique(torch.where(mask, dst, src)[mask])
    weights = probs[neighbors]
    order = torch.argsort(weights, descending=True)
    result = []
    for i in order[:top_k]:
        nid = int(neighbors[i])
        result.append(
            {
                "neighbor_id": nid,
                "prob": float(weights[i]),
            }
        )
    return result


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = EllipticDataset(feature_subset=LOCAL_FEATURES)
    data = dataset.load(verbose=False)
    data = data.to(device)

    ckpt_path = PROJECT_ROOT / "checkpoints" / "graphsage_local_only_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError("graphsage_local_only_best.pt missing. Run scripts/run_m8_graphsage_local_only.py.")

    model = GraphSAGE(
        in_channels=data.x.shape[1],
        hidden_channels=128,
        out_channels=2,
        num_layers=2,
        dropout=0.4,
    ).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    with torch.no_grad():
        logits = model(data.x, data.edge_index)
    top_nodes = select_top_nodes(logits.cpu(), data.test_mask.cpu(), k=5)
    print(f"Explaining nodes: {top_nodes}")

    reports_dir = PROJECT_ROOT / "reports"
    plots_dir = reports_dir / "plots"
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    explanations = []
    for node_id in top_nodes:
        x = data.x.clone().detach()
        x.requires_grad_(True)
        out = model(x, data.edge_index)
        score = out[node_id, 1]
        model.zero_grad()
        score.backward(retain_graph=True)
        grads = x.grad[node_id].detach().cpu().numpy()
        values = x[node_id].detach().cpu().numpy()
        importance = np.abs(grads * values)
        order = np.argsort(importance)[::-1][:10]
        top_feats = [
            {"feature": LOCAL_FEATURES[i], "importance": float(importance[i])}
            for i in order
        ]

        probs = torch.softmax(out.detach(), dim=1)[:, 1].cpu()
        neighbors = neighbor_summary(data.edge_index.cpu(), node_id, probs, top_k=10)

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.barh(
            [LOCAL_FEATURES[i] for i in order[::-1]],
            importance[order[::-1]],
            color="darkorange",
        )
        ax.set_xlabel("|grad * input|")
        ax.set_title(f"Node {node_id} top features")
        plt.tight_layout()
        plot_path = plots_dir / f"m8_graphsage_saliency_node{node_id}.png"
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)

        explanations.append(
            {
                "node_id": node_id,
                "label": int(data.y[node_id].item()),
                "top_features": top_feats,
                "top_neighbors": neighbors,
                "plot_path": str(plot_path),
            }
        )

    out_path = reports_dir / "m8_graphsage_saliency.json"
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(explanations, fp, indent=2)
    print(f"Saved saliency summary to {out_path}")


if __name__ == "__main__":
    main()
