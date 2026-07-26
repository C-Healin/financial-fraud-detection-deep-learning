"""GNN node embeddings combined with XGBoost for downstream classification."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from ..constants import CARD_COLUMN, MERCHANT_COLUMN, TARGET
from ..metrics import classification_metrics, select_threshold
from ..preprocessing import FraudPreprocessor


def _hybrid_matrix(
    df: pd.DataFrame,
    preprocessor: FraudPreprocessor,
    graph_bundle: dict[str, object],
) -> np.ndarray:
    base = preprocessor.transform_tabular(df)
    card_map = graph_bundle["card_map"]
    merchant_map = graph_bundle["merchant_map"]
    embeddings = graph_bundle["embeddings"]
    card_idx = df[CARD_COLUMN].astype(str).map(card_map).to_numpy(dtype=int)
    merchant_idx = df[MERCHANT_COLUMN].astype(str).map(merchant_map).to_numpy(dtype=int)
    return np.concatenate([base, embeddings[card_idx], embeddings[merchant_idx]], axis=1)


def train_hybrid_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    graph_bundle: dict[str, object],
    artifact_dir: str | Path,
) -> tuple[dict[str, object], np.ndarray]:
    artifact_dir = Path(artifact_dir)
    preprocessor = FraudPreprocessor.create().fit(train_df)
    x_train = _hybrid_matrix(train_df, preprocessor, graph_bundle)
    x_val = _hybrid_matrix(val_df, preprocessor, graph_bundle)
    x_test = _hybrid_matrix(test_df, preprocessor, graph_bundle)
    y_train = train_df[TARGET].to_numpy(dtype=int)
    y_val = val_df[TARGET].to_numpy(dtype=int)
    y_test = test_df[TARGET].to_numpy(dtype=int)
    positives = max(int(y_train.sum()), 1)
    negatives = max(len(y_train) - positives, 1)

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=100,
        max_depth=2,
        learning_rate=0.04,
        subsample=0.80,
        colsample_bytree=0.70,
        reg_lambda=5.0,
        reg_alpha=0.2,
        scale_pos_weight=negatives / positives,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    val_probability = model.predict_proba(x_val)[:, 1]
    threshold = select_threshold(y_val, val_probability)
    test_probability = model.predict_proba(x_test)[:, 1]
    metrics = classification_metrics(y_test, test_probability, threshold)
    metrics["model"] = "GNN Embeddings + XGBoost"

    model.save_model(artifact_dir / "hybrid_xgboost_model.json")
    preprocessor.save(artifact_dir / "hybrid_preprocessor.joblib")
    (artifact_dir / "hybrid_metadata.json").write_text(
        json.dumps({"threshold": threshold, "embedding_dim": 48}, indent=2), encoding="utf-8"
    )
    return metrics, test_probability
