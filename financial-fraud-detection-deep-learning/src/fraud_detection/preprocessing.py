"""Leakage-safe preprocessing for tabular and neural-network models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from .constants import CATEGORICAL_FEATURES, NUMERICAL_FEATURES


@dataclass
class FraudPreprocessor:
    categorical_features: list[str]
    numerical_features: list[str]
    encoder: OrdinalEncoder
    scaler: StandardScaler

    @classmethod
    def create(
        cls,
        categorical_features: Iterable[str] = CATEGORICAL_FEATURES,
        numerical_features: Iterable[str] = NUMERICAL_FEATURES,
    ) -> "FraudPreprocessor":
        return cls(
            categorical_features=list(categorical_features),
            numerical_features=list(numerical_features),
            encoder=OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                encoded_missing_value=-1,
                dtype=np.float32,
            ),
            scaler=StandardScaler(),
        )

    def fit(self, df: pd.DataFrame) -> "FraudPreprocessor":
        cats = df[self.categorical_features].astype(str).fillna("__MISSING__")
        nums = df[self.numerical_features].astype(float).fillna(0.0)
        self.encoder.fit(cats)
        self.scaler.fit(nums)
        return self

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        cats = df[self.categorical_features].astype(str).fillna("__MISSING__")
        nums = df[self.numerical_features].astype(float).fillna(0.0)
        # +1 reserves index 0 for unseen/missing categories.
        cat_values = self.encoder.transform(cats).astype(np.int64) + 1
        num_values = self.scaler.transform(nums).astype(np.float32)
        return cat_values, num_values

    def transform_tabular(self, df: pd.DataFrame) -> np.ndarray:
        cats, nums = self.transform(df)
        return np.concatenate([nums, cats.astype(np.float32)], axis=1)

    @property
    def categorical_cardinalities(self) -> list[int]:
        return [len(values) + 1 for values in self.encoder.categories_]

    @property
    def output_feature_names(self) -> list[str]:
        return self.numerical_features + self.categorical_features

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "FraudPreprocessor":
        return joblib.load(path)
