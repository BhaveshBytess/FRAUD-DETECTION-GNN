"""
M5 — Tabular baselines on Elliptic++.

This script is import-friendly: notebooks can reuse the helper functions
without executing the CLI entrypoint. Run directly to train all tabular
models locally and save metrics/plots under reports/.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import xgboost as xgb  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import append_metrics_to_csv, save_metrics_json  # noqa: E402
from src.utils.metrics import (  # noqa: E402
    compute_metrics,
    compute_recall_at_k,
    find_best_f1_threshold,
)
from src.utils.seed import set_all_seeds  # noqa: E402

SEED = 42
sns.set_theme(style="ticks")


@dataclass
class TabularDataset:
    """Containers for scaled feature matrices and metadata."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_cols: List[str]
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    stats: Dict[str, float]
    scaler: StandardScaler


@dataclass
class ModelResult:
    """Stores metrics and score vectors for a trained model."""

    model: str
    metrics: Dict[str, float]
    val_probs: np.ndarray
    test_probs: np.ndarray
    train_seconds: float


def _resolve_timestamp_column(df: pd.DataFrame) -> str:
    for candidate in ("timestamp", "Time step", "time_step"):
        if candidate in df.columns:
            return candidate
    raise KeyError(
        "Could not find timestamp column. Expected one of "
        "`timestamp`, `Time step`, or `time_step`."
    )


def load_tabular_dataset(
    data_dir: Path,
    selected_features: List[str] | None = None,
    train_time_end: int | None = None,
    val_time_end: int | None = None,
) -> TabularDataset:
    """Load Elliptic++ features, apply temporal split, scale features.

    Args:
        data_dir: Directory containing `txs_features.csv` and `splits.json`.
        selected_features: Optional list of feature columns to keep. If None,
            all available feature columns are used.
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")

    splits_path = data_dir / "splits.json"
    if not splits_path.exists():
        raise FileNotFoundError(
            "splits.json not found. Run `python -m src.data.elliptic_loader --check` first."
        )

    with open(splits_path, "r", encoding="utf-8") as fp:
        split_info = json.load(fp)

    # Allow overriding temporal boundaries
    if train_time_end is not None or val_time_end is not None:
        if train_time_end is None or val_time_end is None:
            raise ValueError("Both train_time_end and val_time_end must be provided together.")
        split_info["train_time_end"] = train_time_end
        split_info["val_time_end"] = val_time_end

    features_df = pd.read_csv(data_dir / "txs_features.csv")
    classes_df = pd.read_csv(data_dir / "txs_classes.csv")
    merged = features_df.merge(classes_df, on="txId", how="left")

    ts_col = _resolve_timestamp_column(merged)
    if ts_col != "timestamp":
        merged.rename(columns={ts_col: "timestamp"}, inplace=True)

    labeled = merged[merged["class"].isin([1, 2])].copy()
    labeled["label"] = (labeled["class"] == 1).astype(int)
    labeled = labeled.sort_values("timestamp")

    train_df = labeled[labeled["timestamp"] <= split_info["train_time_end"]].copy()
    val_df = labeled[
        (labeled["timestamp"] > split_info["train_time_end"])
        & (labeled["timestamp"] <= split_info["val_time_end"])
    ].copy()
    test_df = labeled[labeled["timestamp"] > split_info["val_time_end"]].copy()

    for split_name, df in (("Train", train_df), ("Val", val_df), ("Test", test_df)):
        if df.empty:
            raise RuntimeError(f"{split_name} split is empty. Check split boundaries.")

    exclude = {"txId", "timestamp", "Time step", "time_step", "class", "label"}
    feature_cols = [col for col in labeled.columns if col not in exclude]
    if selected_features is not None:
        missing = sorted(set(selected_features) - set(feature_cols))
        if missing:
            raise ValueError(
                "Selected features missing from dataset: "
                + ", ".join(missing[:10])
                + ("..." if len(missing) > 10 else "")
            )
        feature_cols = [col for col in feature_cols if col in selected_features]
        if not feature_cols:
            raise ValueError("No overlapping features after applying selection list.")

    def _values(frame: pd.DataFrame) -> np.ndarray:
        arr = frame[feature_cols].to_numpy(dtype=np.float32)
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    X_train = _values(train_df)
    X_val = _values(val_df)
    X_test = _values(test_df)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    stats = {
        "train_count": len(train_df),
        "val_count": len(val_df),
        "test_count": len(test_df),
        "train_fraud_pct": float(train_df["label"].mean() * 100),
        "val_fraud_pct": float(val_df["label"].mean() * 100),
        "test_fraud_pct": float(test_df["label"].mean() * 100),
        "train_time_end": split_info["train_time_end"],
        "val_time_end": split_info["val_time_end"],
        "feature_count": len(feature_cols),
    }

    print("\n[Data] Temporal split summary (labeled transactions only)")
    for split_name, count_key, fraud_key in [
        ("Train", "train_count", "train_fraud_pct"),
        ("Val", "val_count", "val_fraud_pct"),
        ("Test", "test_count", "test_fraud_pct"),
    ]:
        print(
            f"  {split_name:<5}: {stats[count_key]:>6} nodes | "
            f"Fraud={stats[fraud_key]:5.2f}%"
        )

    return TabularDataset(
        X_train=X_train,
        y_train=train_df["label"].to_numpy(),
        X_val=X_val,
        y_val=val_df["label"].to_numpy(),
        X_test=X_test,
        y_test=test_df["label"].to_numpy(),
        feature_cols=feature_cols,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        stats=stats,
        scaler=scaler,
    )


def subset_tabular_dataset(
    base: TabularDataset,
    selected_features: List[str],
) -> TabularDataset:
    """Create a new TabularDataset with a subset of features (re-scaling data)."""
    if not selected_features:
        raise ValueError("selected_features must contain at least one column.")

    base_feature_set = set(base.feature_cols)
    missing = sorted(set(selected_features) - base_feature_set)
    if missing:
        raise ValueError(
            "Selected features missing from base dataset: "
            + ", ".join(missing[:10])
            + ("..." if len(missing) > 10 else "")
        )

    feature_cols = [col for col in base.feature_cols if col in selected_features]
    if not feature_cols:
        raise ValueError("No overlapping features after applying selection list.")

    def _values(frame: pd.DataFrame) -> np.ndarray:
        arr = frame[feature_cols].to_numpy(dtype=np.float32)
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(_values(base.train_df))
    X_val = scaler.transform(_values(base.val_df))
    X_test = scaler.transform(_values(base.test_df))

    stats = base.stats.copy()
    stats["feature_count"] = len(feature_cols)

    return TabularDataset(
        X_train=X_train,
        y_train=base.y_train,
        X_val=X_val,
        y_val=base.y_val,
        X_test=X_test,
        y_test=base.y_test,
        feature_cols=feature_cols,
        train_df=base.train_df,
        val_df=base.val_df,
        test_df=base.test_df,
        stats=stats,
        scaler=scaler,
    )


def _evaluate_predictions(
    model_name: str,
    y_val: np.ndarray,
    val_probs: np.ndarray,
    y_test: np.ndarray,
    test_probs: np.ndarray,
) -> Dict[str, float]:
    threshold, best_f1 = find_best_f1_threshold(y_val, val_probs)
    metrics = compute_metrics(y_test, test_probs, threshold=threshold)
    recall_dict = compute_recall_at_k(y_test, test_probs, k_fracs=[0.01])
    metrics.update(
        {
            "model": model_name,
            "best_val_f1": float(best_f1),
            "recall@1.0%": float(recall_dict["recall@1.0%"]),
        }
    )
    # Ensure all numeric values are native Python floats for JSON serialization.
    clean_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, (np.floating, np.integer)):
            clean_metrics[key] = float(value)
        else:
            clean_metrics[key] = value
    return clean_metrics


def train_logistic_regression(data: TabularDataset) -> ModelResult:
    start = time.perf_counter()
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(data.X_train, data.y_train)
    val_probs = model.predict_proba(data.X_val)[:, 1]
    test_probs = model.predict_proba(data.X_test)[:, 1]
    metrics = _evaluate_predictions("Logistic Regression", data.y_val, val_probs, data.y_test, test_probs)
    elapsed = time.perf_counter() - start
    print(
        f"[LR] PR-AUC={metrics['pr_auc']:.4f} | ROC-AUC={metrics['roc_auc']:.4f} | "
        f"F1={metrics['f1']:.4f} | t={elapsed:.1f}s"
    )
    return ModelResult("Logistic Regression", metrics, val_probs, test_probs, elapsed)


def train_random_forest(data: TabularDataset) -> ModelResult:
    start = time.perf_counter()
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=18,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=4,
        random_state=SEED,
    )
    model.fit(data.X_train, data.y_train)
    val_probs = model.predict_proba(data.X_val)[:, 1]
    test_probs = model.predict_proba(data.X_test)[:, 1]
    metrics = _evaluate_predictions("Random Forest", data.y_val, val_probs, data.y_test, test_probs)
    elapsed = time.perf_counter() - start
    print(
        f"[RF] PR-AUC={metrics['pr_auc']:.4f} | ROC-AUC={metrics['roc_auc']:.4f} | "
        f"F1={metrics['f1']:.4f} | t={elapsed:.1f}s"
    )
    return ModelResult("Random Forest", metrics, val_probs, test_probs, elapsed)


def train_xgboost(data: TabularDataset) -> ModelResult:
    start = time.perf_counter()
    scale_pos_weight = (data.y_train == 0).sum() / max((data.y_train == 1).sum(), 1)
    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=SEED,
        tree_method="hist",
        eval_metric="aucpr",
        n_jobs=-1,
        early_stopping_rounds=20,
    )
    model.fit(
        data.X_train,
        data.y_train,
        eval_set=[(data.X_val, data.y_val)],
        verbose=False,
    )
    val_probs = model.predict_proba(data.X_val)[:, 1]
    test_probs = model.predict_proba(data.X_test)[:, 1]
    metrics = _evaluate_predictions("XGBoost", data.y_val, val_probs, data.y_test, test_probs)
    elapsed = time.perf_counter() - start
    print(
        f"[XGB] PR-AUC={metrics['pr_auc']:.4f} | ROC-AUC={metrics['roc_auc']:.4f} | "
        f"F1={metrics['f1']:.4f} | t={elapsed:.1f}s"
    )
    return ModelResult("XGBoost", metrics, val_probs, test_probs, elapsed)


def train_mlp(data: TabularDataset) -> ModelResult:
    start = time.perf_counter()
    pos_weight = (data.y_train == 0).sum() / max((data.y_train == 1).sum(), 1)
    sample_weight = np.where(data.y_train == 1, pos_weight, 1.0)
    model = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        learning_rate="adaptive",
        learning_rate_init=1e-3,
        batch_size=512,
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=SEED,
    )
    model.fit(data.X_train, data.y_train, sample_weight=sample_weight)
    val_probs = model.predict_proba(data.X_val)[:, 1]
    test_probs = model.predict_proba(data.X_test)[:, 1]
    metrics = _evaluate_predictions("MLP", data.y_val, val_probs, data.y_test, test_probs)
    elapsed = time.perf_counter() - start
    print(
        f"[MLP] PR-AUC={metrics['pr_auc']:.4f} | ROC-AUC={metrics['roc_auc']:.4f} | "
        f"F1={metrics['f1']:.4f} | t={elapsed:.1f}s"
    )
    return ModelResult("MLP", metrics, val_probs, test_probs, elapsed)


def summarize_results(results: List[ModelResult]) -> pd.DataFrame:
    """Convert model results to a DataFrame sorted by PR-AUC."""
    df = pd.DataFrame([res.metrics for res in results])
    return df.sort_values("pr_auc", ascending=False)


def _model_filename(model_name: str) -> str:
    return (
        model_name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        + "_metrics.json"
    )


def save_artifacts(
    results: List[ModelResult],
    y_test: np.ndarray,
    reports_dir: Path,
    experiment_name: str = "elliptic-gnn-baselines",
):
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = reports_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary_path = reports_dir / "metrics_summary.csv"

    for res in results:
        metrics = res.metrics.copy()
        save_metrics_json(metrics, reports_dir / _model_filename(res.model))
        append_metrics_to_csv(
            metrics,
            filepath=summary_path,
            experiment_name=experiment_name,
            model_name=res.model,
            split="test",
        )

    df = summarize_results(results)
    comparison_csv = reports_dir / "all_models_comparison.csv"
    df.to_csv(comparison_csv, index=False)

    print(f"\n[Artifacts] Saved comparison CSV to {comparison_csv}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric, title in zip(
        axes,
        ["pr_auc", "roc_auc", "f1"],
        ["PR-AUC (Primary)", "ROC-AUC", "F1 (val threshold)"],
    ):
        ordered = df.sort_values(metric, ascending=True)
        ax.barh(ordered["model"], ordered[metric], color="steelblue", alpha=0.8)
        ax.set_xlabel(metric.upper())
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.3, linestyle="--", linewidth=0.5)
        for idx, value in enumerate(ordered[metric]):
            ax.text(value + 0.005, idx, f"{value:.3f}", va="center", fontsize=8)
    plt.tight_layout()
    comp_plot = plots_dir / "all_models_comparison.png"
    plt.savefig(comp_plot, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[Artifacts] Saved comparison plot to {comp_plot}")

    best_result = max(results, key=lambda r: r.metrics["pr_auc"])
    precision, recall, _ = compute_recall_curve(y_test, best_result.test_probs)
    fpr, tpr, _ = compute_roc_curve(y_test, best_result.test_probs)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(recall, precision, label=f"{best_result.model} (AP={best_result.metrics['pr_auc']:.3f})")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_title("Precision-Recall Curve")
    axes[0].grid(alpha=0.3)

    axes[1].plot(fpr, tpr, label=f"{best_result.model} (ROC-AUC={best_result.metrics['roc_auc']:.3f})")
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve")
    axes[1].grid(alpha=0.3)

    for ax in axes:
        ax.legend(loc="lower right")
    plt.tight_layout()
    curve_plot = plots_dir / f"{best_result.model.lower().replace(' ', '_')}_pr_roc.png"
    plt.savefig(curve_plot, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[Artifacts] Saved PR/ROC curves to {curve_plot}")


def compute_recall_curve(y_true: np.ndarray, y_scores: np.ndarray):
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    return precision, recall, thresholds


def compute_roc_curve(y_true: np.ndarray, y_scores: np.ndarray):
    from sklearn.metrics import roc_curve

    return roc_curve(y_true, y_scores)


def run_all_models(data: TabularDataset) -> List[ModelResult]:
    """Convenience helper used by notebooks."""
    return [
        train_logistic_regression(data),
        train_random_forest(data),
        train_xgboost(data),
        train_mlp(data),
    ]


def main():
    set_all_seeds(SEED)
    data_dir = PROJECT_ROOT / "data" / "Elliptic++ Dataset"
    reports_dir = PROJECT_ROOT / "reports"

    print("=" * 80)
    print("M5 — TABULAR BASELINES (LOCAL)")
    print("=" * 80)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Data directory: {data_dir}")
    print(f"Reports dir  : {reports_dir}")
    print("=" * 80)

    data = load_tabular_dataset(data_dir)
    results = run_all_models(data)

    df = summarize_results(results)
    print("\nFinal Tabular Results (sorted by PR-AUC)")
    print(df[["model", "pr_auc", "roc_auc", "f1", "recall@1.0%", "threshold"]])

    save_artifacts(results, data.y_test, reports_dir)

    print("\n[Next] Compare against GNN baselines after they are retrained with the corrected labels.")


if __name__ == "__main__":
    main()
