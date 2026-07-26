"""Production-style inference wrapper used by FastAPI and Streamlit."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .constants import TABULAR_FEATURES
from .models.xgboost_model import load_xgboost_artifacts


class FraudPredictor:
    def __init__(self, artifact_dir: str | Path) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.preprocessor, self.model, self.metadata = load_xgboost_artifacts(self.artifact_dir)
        self.threshold = float(self.metadata["threshold"])

    def predict_one(self, transaction: dict[str, object]) -> dict[str, object]:
        missing = [feature for feature in TABULAR_FEATURES if feature not in transaction]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        frame = pd.DataFrame([{feature: transaction[feature] for feature in TABULAR_FEATURES}])
        matrix = self.preprocessor.transform_tabular(frame)
        probability = float(self.model.predict_proba(matrix)[0, 1])
        risk_level = "high" if probability >= self.threshold else "medium" if probability >= self.threshold * 0.55 else "low"
        return {
            "fraud_probability": round(probability, 6),
            "fraud_prediction": int(probability >= self.threshold),
            "risk_level": risk_level,
            "decision_threshold": round(self.threshold, 6),
            "model": "xgboost",
        }
