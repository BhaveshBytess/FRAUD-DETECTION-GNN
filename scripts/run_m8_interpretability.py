"""
M8 — Interpretability helper script.

Computes tree-SHAP (using XGBoost's built-in pred_contribs) for the full-feature
tabular model and saves:
  - booster checkpoint (`reports/models/xgb_full.json`)
  - global importance CSV (`reports/m8_xgb_shap_importance.csv`)
  - horizontal bar plot (`reports/plots/m8_xgb_shap_summary.png`)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_m5_tabular import load_tabular_dataset, set_all_seeds

SEED = 42


def train_xgboost_full(data_dir: Path) -> Dict[str, object]:
    data = load_tabular_dataset(data_dir)
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
    return {"model": model, "data": data}


def compute_shap(model: xgb.XGBClassifier, data, split: str = "test") -> pd.DataFrame:
    booster = model.get_booster()
    feature_names = data.feature_cols
    matrix = xgb.DMatrix(
        data.X_test if split == "test" else data.X_val,
        feature_names=feature_names,
    )
    contribs = booster.predict(matrix, pred_contribs=True)
    contribs = np.asarray(contribs)
    feature_contribs = contribs[:, :-1]  # drop bias column
    df = pd.DataFrame(feature_contribs, columns=feature_names)
    summary = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_contribution": df.abs().mean(axis=0).values,
            "mean_signed_contribution": df.mean(axis=0).values,
        }
    ).sort_values("mean_abs_contribution", ascending=False)
    return summary


def save_artifacts(
    model: xgb.XGBClassifier,
    shap_df: pd.DataFrame,
    reports_dir: Path,
    models_dir: Path,
) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = reports_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "xgb_full.json"
    model.save_model(model_path)

    shap_csv = reports_dir / "m8_xgb_shap_importance.csv"
    shap_df.to_csv(shap_csv, index=False)

    top_k = shap_df.head(25)
    plt.figure(figsize=(8, max(6, len(top_k) * 0.3)))
    plt.barh(top_k["feature"][::-1], top_k["mean_abs_contribution"][::-1], color="steelblue")
    plt.xlabel("Mean |SHAP| (log-odds)")
    plt.title("XGBoost (Full Features) — Global SHAP Importance")
    plt.tight_layout()
    plot_path = plots_dir / "m8_xgb_shap_summary.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    summary_json = {
        "model_path": str(model_path),
        "shap_csv": str(shap_csv),
        "plot_path": str(plot_path),
    }
    with open(reports_dir / "m8_xgb_shap_artifacts.json", "w", encoding="utf-8") as fp:
        json.dump(summary_json, fp, indent=2)


def main() -> None:
    set_all_seeds(SEED)
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data" / "Elliptic++ Dataset"
    reports_dir = project_root / "reports"
    models_dir = reports_dir / "models"

    print("=" * 80)
    print("M8 — XGBoost SHAP Summary (full features)")
    print("=" * 80)

    result = train_xgboost_full(data_dir)
    model = result["model"]
    data = result["data"]

    shap_df = compute_shap(model, data, split="test")
    save_artifacts(model, shap_df, reports_dir, models_dir)

    print("\nTop 10 features by mean |SHAP|:")
    print(shap_df.head(10))


if __name__ == "__main__":
    main()
