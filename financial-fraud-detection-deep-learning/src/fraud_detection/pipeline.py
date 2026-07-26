"""End-to-end training orchestration."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .constants import ID_COLUMN, TARGET
from .data import load_dataset, split_dataset
from .metrics import create_comparison_reports, save_model_metrics, save_prediction_file
from .models.gnn import train_gnn_model
from .models.hybrid import train_hybrid_model
from .models.mlp import train_mlp_model
from .models.xgboost_model import train_xgboost_model


def run_training_pipeline(data_path: str | Path, project_root: str | Path) -> dict[str, dict[str, object]]:
    project_root = Path(project_root)
    artifact_dir = project_root / "artifacts" / "models"
    prediction_dir = project_root / "artifacts" / "predictions"
    report_dir = project_root / "reports"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(data_path)
    train_df, val_df, test_df = split_dataset(df)
    for name, split in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        split.to_csv(project_root / "data" / "processed" / f"{name}.csv", index=False)

    metrics_by_model: dict[str, dict[str, object]] = {}
    predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    y_test = test_df[TARGET].to_numpy(dtype=int)

    xgb_metrics, xgb_prob, _, _ = train_xgboost_model(train_df, val_df, test_df, artifact_dir)
    metrics_by_model["XGBoost"] = xgb_metrics
    predictions["XGBoost"] = (y_test, xgb_prob)

    mlp_metrics, mlp_prob, _, _ = train_mlp_model(train_df, val_df, test_df, artifact_dir)
    metrics_by_model["Embedding MLP"] = mlp_metrics
    predictions["Embedding MLP"] = (y_test, mlp_prob)

    gnn_metrics, gnn_prob, graph_bundle = train_gnn_model(train_df, val_df, test_df, artifact_dir)
    metrics_by_model["GraphSAGE"] = gnn_metrics
    predictions["GraphSAGE"] = (y_test, gnn_prob)

    hybrid_metrics, hybrid_prob = train_hybrid_model(
        train_df, val_df, test_df, graph_bundle, artifact_dir
    )
    metrics_by_model["GNN + XGBoost"] = hybrid_metrics
    predictions["GNN + XGBoost"] = (y_test, hybrid_prob)

    for name, probability in {
        "xgboost": xgb_prob,
        "mlp": mlp_prob,
        "gnn": gnn_prob,
        "gnn_xgboost": hybrid_prob,
    }.items():
        save_prediction_file(
            test_df[ID_COLUMN], y_test, probability, prediction_dir / f"{name}.csv"
        )

    save_model_metrics(metrics_by_model, report_dir / "model_metrics.json")
    create_comparison_reports(predictions, metrics_by_model, report_dir)
    return metrics_by_model
