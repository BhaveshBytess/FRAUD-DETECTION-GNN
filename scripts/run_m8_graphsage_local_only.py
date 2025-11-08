"""
M8 helper — trains GraphSAGE on local-only features (AF1–AF93) so we can run
downstream interpretability (GNNExplainer) without relying on Kaggle artifacts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.elliptic_loader import EllipticDataset
from src.data.feature_groups import LOCAL_FEATURES
from src.models.graphsage import GraphSAGE, GraphSAGETrainer


def main() -> None:
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print("M8 — GraphSAGE (local-only features)")
    print("=" * 80)
    print(f"Device: {device}")

    dataset = EllipticDataset(feature_subset=LOCAL_FEATURES)
    data = dataset.load(verbose=True)
    data = data.to(device)
    data.train_mask = data.train_mask.to(device)
    data.val_mask = data.val_mask.to(device)
    data.test_mask = data.test_mask.to(device)

    model = GraphSAGE(
        in_channels=data.x.shape[1],
        hidden_channels=128,
        out_channels=2,
        num_layers=2,
        dropout=0.4,
    )
    trainer = GraphSAGETrainer(
        model=model,
        data=data,
        device=device,
        lr=1e-3,
        weight_decay=5e-4,
    )
    history = trainer.fit(epochs=60, patience=10, eval_metric="pr_auc", verbose=True)
    test_metrics = trainer.test(data.test_mask)

    reports_dir = PROJECT_ROOT / "reports"
    checkpoints_dir = PROJECT_ROOT / "checkpoints"
    reports_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    metrics_clean = {k: float(v) for k, v in test_metrics.items()}

    metrics_path = reports_dir / "graphsage_local_only_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as fp:
        json.dump(metrics_clean, fp, indent=2)

    ckpt_path = checkpoints_dir / "graphsage_local_only_best.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "hidden_channels": 128,
                "num_layers": 2,
                "dropout": 0.4,
                "feature_subset": LOCAL_FEATURES,
            },
            "metrics": metrics_clean,
            "history": history,
        },
        ckpt_path,
    )

    print("\nTest metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"\nSaved metrics to {metrics_path}")
    print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
