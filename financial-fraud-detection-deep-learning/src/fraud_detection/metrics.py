"""Fraud-focused metrics, threshold selection, and report plots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def select_threshold(y_true: np.ndarray, probability: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probability)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def classification_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> dict[str, float | int | list[list[int]]]:
    prediction = (probability >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "threshold": float(threshold),
        "fraud_rate": float(np.mean(y_true)),
        "support": int(len(y_true)),
        "confusion_matrix": confusion_matrix(y_true, prediction).tolist(),
    }


def save_prediction_file(
    transaction_ids: pd.Series,
    y_true: np.ndarray,
    probability: np.ndarray,
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "transaction_id": transaction_ids.astype(str).to_numpy(),
            "is_fraud": y_true.astype(int),
            "fraud_probability": probability.astype(float),
        }
    ).to_csv(path, index=False)


def save_model_metrics(metrics: Mapping[str, object], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def create_comparison_reports(
    predictions: Mapping[str, tuple[np.ndarray, np.ndarray]],
    metrics_by_model: Mapping[str, Mapping[str, float]],
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, (y_true, probability) in predictions.items():
        fpr, tpr, _ = roc_curve(y_true, probability)
        auc = roc_auc_score(y_true, probability)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", label="No skill")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curve.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, (y_true, probability) in predictions.items():
        precision, recall, _ = precision_recall_curve(y_true, probability)
        ap = average_precision_score(y_true, probability)
        ax.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
    baseline = float(np.mean(next(iter(predictions.values()))[0]))
    ax.axhline(baseline, linestyle="--", label=f"Fraud rate={baseline:.3f}")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision–Recall Curve")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "pr_curve.png", dpi=180)
    plt.close(fig)

    names = list(metrics_by_model)
    pr_auc = [float(metrics_by_model[name]["pr_auc"]) for name in names]
    roc_auc = [float(metrics_by_model[name]["roc_auc"]) for name in names]
    x = np.arange(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, pr_auc, width, label="PR-AUC")
    ax.bar(x + width / 2, roc_auc, width, label="ROC-AUC")
    ax.set_xticks(x, names, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.legend()
    for idx, value in enumerate(pr_auc):
        ax.text(idx - width / 2, value + 0.015, f"{value:.3f}", ha="center", fontsize=8)
    for idx, value in enumerate(roc_auc):
        ax.text(idx + width / 2, value + 0.015, f"{value:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "model_comparison.png", dpi=180)
    plt.close(fig)
