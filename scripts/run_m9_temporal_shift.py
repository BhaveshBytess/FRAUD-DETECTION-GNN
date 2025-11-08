"""
M9 — Temporal robustness study.

Trains XGBoost (full features) and GraphSAGE (local-only) under multiple
train/val/test temporal boundaries to observe degradation when testing farther
into the future.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_m5_tabular import load_tabular_dataset, set_all_seeds, train_xgboost  # type: ignore
from src.data.elliptic_loader import EllipticDataset  # type: ignore
from src.data.feature_groups import LOCAL_FEATURES  # type: ignore
from src.models.graphsage import GraphSAGE, GraphSAGETrainer  # type: ignore

SCENARIOS = [
    {"name": "baseline", "train_end": 29, "val_end": 39},
    {"name": "shift_mid", "train_end": 24, "val_end": 34},
    {"name": "shift_long", "train_end": 20, "val_end": 30},
]


def train_graphsage_local(train_end: int, val_end: int, device: str) -> Dict[str, float]:
    dataset = EllipticDataset(
        feature_subset=LOCAL_FEATURES,
        train_time_end=train_end,
        val_time_end=val_end,
    )
    data = dataset.load(verbose=False)
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
    trainer.fit(epochs=50, patience=8, eval_metric="pr_auc", verbose=False)
    metrics = trainer.test(data.test_mask)
    return {k: float(v) for k, v in metrics.items()}


def main() -> None:
    set_all_seeds(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = PROJECT_ROOT / "data" / "Elliptic++ Dataset"
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, object]] = []

    for scenario in SCENARIOS:
        print("=" * 80)
        print(
            f"Scenario {scenario['name']} | "
            f"train_end={scenario['train_end']} | val_end={scenario['val_end']}"
        )
        print("=" * 80)

        tab_data = load_tabular_dataset(
            data_dir,
            train_time_end=scenario["train_end"],
            val_time_end=scenario["val_end"],
        )
        xgb_result = train_xgboost(tab_data)
        results.append(
            {
                "scenario": scenario["name"],
                "model": "XGBoost_full",
                **xgb_result.metrics,
            }
        )

        gs_metrics = train_graphsage_local(scenario["train_end"], scenario["val_end"], device)
        results.append(
            {
                "scenario": scenario["name"],
                "model": "GraphSAGE_local",
                **gs_metrics,
            }
        )

    df = pd.DataFrame(results)
    out_csv = reports_dir / "m9_temporal_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved temporal robustness results to {out_csv}")


if __name__ == "__main__":
    main()
