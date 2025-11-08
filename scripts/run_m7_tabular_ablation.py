"""
M7 — Tabular feature ablation experiments (local only vs aggregate only, etc.).

This script focuses on the ML baselines (LR/RF/XGB/MLP) which we can execute
locally. Each configuration defines a specific feature subset so we can test the
feature-dominance hypothesis without requiring GNN retraining.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import gc  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_m5_tabular import (  # noqa: E402
    load_tabular_dataset,
    run_all_models,
    subset_tabular_dataset,
)
from src.data.feature_groups import (  # noqa: E402
    AGGREGATE_FEATURES,
    LOCAL_FEATURES,
    STRUCTURAL_FEATURES,
)
from src.utils.logger import append_metrics_to_csv  # noqa: E402
from src.utils.seed import set_all_seeds  # noqa: E402

CONFIGS: Dict[str, Dict[str, object]] = {
    "full": {
        "features": None,
        "description": "All features (baseline reference)",
    },
    "local_only": {
        "features": LOCAL_FEATURES,
        "description": "AF1–AF93 local transaction descriptors",
    },
    "aggregate_only": {
        "features": AGGREGATE_FEATURES,
        "description": "AF94–AF182 neighbor-aggregated descriptors",
    },
    "local_plus_structural": {
        "features": LOCAL_FEATURES + STRUCTURAL_FEATURES,
        "description": "Local descriptors plus manual structural stats",
    },
    "structural_only": {
        "features": STRUCTURAL_FEATURES,
        "description": "Hand-crafted structural statistics only",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run M7 tabular ablation configs (local vs aggregate feature sets)."
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=sorted(CONFIGS.keys()),
        default=["full", "local_only", "aggregate_only", "local_plus_structural"],
        help="Subset of configurations to run. Default runs the three primary configs plus local+structural.",
    )
    parser.add_argument(
        "--reports-dir",
        default=str(PROJECT_ROOT / "reports"),
        help="Directory for saving CSV summaries and plots.",
    )
    parser.add_argument(
        "--data-dir",
        default=str(PROJECT_ROOT / "data" / "Elliptic++ Dataset"),
        help="Directory containing txs_features.csv and splits.json.",
    )
    return parser.parse_args()


def run_configuration(
    name: str,
    config: Dict[str, object],
    dataset,
    reports_dir: Path,
    release_dataset: bool,
) -> List[Dict[str, object]]:
    """Load dataset with the requested features and run all tabular models."""
    print("\n" + "=" * 80)
    print(f"M7 — CONFIG: {name}")
    print("=" * 80)
    print(f"Description : {config['description']}")
    if config["features"] is None:
        print("Feature set : full (all columns)")
    else:
        print(f"Feature set : {len(config['features'])} columns")
    feature_count = len(dataset.feature_cols)
    print(f"Effective feature count after filtering: {feature_count}")

    results = run_all_models(dataset)
    rows: List[Dict[str, object]] = []
    summary_csv = reports_dir / "metrics_summary.csv"

    for res in results:
        row = {
            "config": name,
            "model": res.model,
            "feature_count": feature_count,
            "config_description": config["description"],
            **res.metrics,
        }
        rows.append(row)
        append_metrics_to_csv(
            res.metrics,
            filepath=summary_csv,
            experiment_name=f"m7-tabular-{name}",
            model_name=res.model,
            split="test",
        )

    if release_dataset:
        # Explicitly free large arrays before the next configuration.
        del dataset
        gc.collect()

    return rows


def save_ablation_results(df_new: pd.DataFrame, reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_csv = reports_dir / "m7_tabular_ablation.csv"
    if out_csv.exists():
        existing = pd.read_csv(out_csv)
        combined = pd.concat([existing, df_new], ignore_index=True)
        combined.sort_values(["config", "model", "pr_auc"], ascending=[True, True, False], inplace=True)
        combined = combined.drop_duplicates(subset=["config", "model"], keep="first")
    else:
        combined = df_new
    combined.to_csv(out_csv, index=False)
    print(f"\n[Artifacts] Saved CSV summary to {out_csv}")

    plot_path = reports_dir / "plots" / "m7_tabular_ablation_pr_auc.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    pivot = combined.pivot_table(index="config", columns="model", values="pr_auc")
    pivot = pivot.sort_index()
    ax = pivot.plot(kind="bar", figsize=(10, 5), colormap="viridis")
    ax.set_ylabel("PR-AUC (test)")
    ax.set_title("M7 Tabular Ablation — PR-AUC by Config and Model")
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[Artifacts] Saved PR-AUC plot to {plot_path}")


def main():
    args = parse_args()
    set_all_seeds(42)

    data_dir = Path(args.data_dir)
    reports_dir = Path(args.reports_dir)

    print("=" * 80)
    print("M7 — TABULAR FEATURE ABLATION")
    print("=" * 80)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Data dir     : {data_dir}")
    print(f"Reports dir  : {reports_dir}")
    print(f"Configs      : {', '.join(args.configs)}")

    base_dataset = load_tabular_dataset(data_dir)
    all_rows: List[Dict[str, object]] = []
    for name in args.configs:
        config = CONFIGS[name]
        if config["features"] is None:
            dataset = base_dataset
            release = False
        else:
            dataset = subset_tabular_dataset(base_dataset, config["features"])
            release = True

        rows = run_configuration(name, config, dataset, reports_dir, release_dataset=release)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.sort_values(["config", "pr_auc"], ascending=[True, False], inplace=True)
    save_ablation_results(df, reports_dir)

    print("\nCompleted configs:")
    for name in args.configs:
        subset = df[df["config"] == name]
        best = subset.iloc[0]
        print(
            f"  {name:<22} | best model={best['model']:<18} | PR-AUC={best['pr_auc']:.3f}"
        )


if __name__ == "__main__":
    main()
