"""XGBoost baseline model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from ..constants import TARGET
from ..metrics import classification_metrics, select_threshold
from ..preprocessing import FraudPreprocessor


def train_xgboost_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    artifact_dir: str | Path,
    use_gpu: bool = False,
) -> tuple[dict[str, object], np.ndarray, FraudPreprocessor, xgb.XGBClassifier]:
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    preprocessor = FraudPreprocessor.create().fit(train_df)
    x_train = preprocessor.transform_tabular(train_df)
    x_val = preprocessor.transform_tabular(val_df)
    x_test = preprocessor.transform_tabular(test_df)
    y_train = train_df[TARGET].to_numpy(dtype=int)
    y_val = val_df[TARGET].to_numpy(dtype=int)
    y_test = test_df[TARGET].to_numpy(dtype=int)

    negatives = max(int((y_train == 0).sum()), 1)
    positives = max(int((y_train == 1).sum()), 1)
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=180,
        max_depth=5,
        learning_rate=0.07,
        subsample=0.82,
        colsample_bytree=0.78,
        min_child_weight=3,
        reg_lambda=1.0,
        scale_pos_weight=negatives / positives,
        tree_method="hist",
        device="cuda" if use_gpu else "cpu",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)

    val_probability = model.predict_proba(x_val)[:, 1]
    threshold = select_threshold(y_val, val_probability)
    test_probability = model.predict_proba(x_test)[:, 1]
    metrics = classification_metrics(y_test, test_probability, threshold)
    metrics["model"] = "XGBoost"

    model.save_model(artifact_dir / "xgboost_model.json")
    preprocessor.save(artifact_dir / "xgboost_preprocessor.joblib")
    (artifact_dir / "xgboost_metadata.json").write_text(
        json.dumps(
            {
                "threshold": threshold,
                "feature_names": preprocessor.output_feature_names,
                "model_type": "xgboost_json",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return metrics, test_probability, preprocessor, model


def load_xgboost_artifacts(artifact_dir: str | Path):
    artifact_dir = Path(artifact_dir)
    preprocessor = FraudPreprocessor.load(artifact_dir / "xgboost_preprocessor.joblib")
    model = xgb.XGBClassifier()
    model.load_model(artifact_dir / "xgboost_model.json")
    metadata = json.loads((artifact_dir / "xgboost_metadata.json").read_text(encoding="utf-8"))
    return preprocessor, model, metadata
