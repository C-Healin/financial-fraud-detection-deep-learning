"""Dataset generation, validation, loading, and deterministic splitting."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .constants import (
    CARD_COLUMN,
    CATEGORICAL_FEATURES,
    ID_COLUMN,
    NUMERICAL_FEATURES,
    RANDOM_STATE,
    TARGET,
)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def generate_synthetic_fraud_data(
    n_transactions: int = 12_000,
    n_cards: int = 900,
    n_merchants: int = 220,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Create a reproducible synthetic transaction dataset.

    The generator deliberately contains both tabular risk patterns and a small
    card--merchant fraud-ring signal. It is safe to publish and is intended for
    demos/tests only; it must not be presented as real customer data.
    """
    if n_transactions < 1_000:
        raise ValueError("n_transactions must be at least 1,000")

    rng = np.random.default_rng(seed)
    cards = np.array([f"card_{i:04d}" for i in range(n_cards)])
    merchants = np.array([f"merchant_{i:03d}" for i in range(n_merchants)])
    jobs = np.array([f"job_{i:02d}" for i in range(32)])
    categories = np.array(
        [
            "grocery_pos",
            "gas_transport",
            "shopping_net",
            "shopping_pos",
            "entertainment",
            "food_dining",
            "travel",
            "health_fitness",
            "personal_care",
            "misc_net",
        ]
    )
    cities = np.array([f"city_{i:02d}" for i in range(60)])
    states = np.array([f"S{i:02d}" for i in range(15)])

    card_profile = pd.DataFrame(
        {
            CARD_COLUMN: cards,
            "gender": rng.choice(["F", "M"], n_cards),
            "job": rng.choice(jobs, n_cards),
            "city": rng.choice(cities, n_cards),
            "state": rng.choice(states, n_cards),
            "zip": rng.integers(10000, 99999, n_cards).astype(str),
            "age": np.clip(rng.normal(44, 14, n_cards), 18, 90).round(0),
            "city_pop": np.maximum(rng.lognormal(10.1, 1.0, n_cards), 800).round(0),
            "card_risk": rng.beta(1.2, 9.0, n_cards),
        }
    ).set_index(CARD_COLUMN)

    merchant_category = rng.choice(categories, n_merchants)
    merchant_risk = rng.beta(1.1, 12.0, n_merchants)
    merchant_base_amt = {
        "grocery_pos": 45,
        "gas_transport": 62,
        "shopping_net": 115,
        "shopping_pos": 95,
        "entertainment": 70,
        "food_dining": 52,
        "travel": 280,
        "health_fitness": 85,
        "personal_care": 48,
        "misc_net": 130,
    }

    ring_cards = set(rng.choice(cards, size=max(20, n_cards // 18), replace=False))
    ring_merchants = set(
        rng.choice(merchants, size=max(10, n_merchants // 15), replace=False)
    )

    sampled_cards = rng.choice(cards, n_transactions)
    sampled_merchants = rng.choice(merchants, n_transactions)
    merchant_idx = np.array([int(x.split("_")[-1]) for x in sampled_merchants])
    sampled_categories = merchant_category[merchant_idx]

    card_rows = card_profile.loc[sampled_cards].reset_index(drop=True)
    base_amount = np.array([merchant_base_amt[x] for x in sampled_categories])
    amount = np.maximum(1.0, rng.lognormal(np.log(base_amount), 0.72))
    distance = rng.gamma(shape=1.5, scale=18.0, size=n_transactions)

    # Inject unusual behaviour that should be detectable by tabular models.
    unusual = rng.random(n_transactions) < 0.045
    amount[unusual] *= rng.uniform(3.0, 9.0, unusual.sum())
    distance[unusual] += rng.uniform(80, 900, unusual.sum())

    start = np.datetime64("2025-01-01T00:00:00")
    seconds = rng.integers(0, 365 * 24 * 3600, n_transactions)
    timestamps = pd.to_datetime(start + seconds.astype("timedelta64[s]"))

    hour = timestamps.hour.to_numpy()
    ring_signal = np.array(
        [c in ring_cards and m in ring_merchants for c, m in zip(sampled_cards, sampled_merchants)],
        dtype=float,
    )
    category_risk = np.isin(sampled_categories, ["shopping_net", "misc_net", "travel"]).astype(float)
    night = ((hour <= 4) | (hour >= 23)).astype(float)
    high_amount = np.log1p(amount) - np.log(80)
    far_distance = np.log1p(distance) - np.log(25)

    card_risk = card_rows["card_risk"].to_numpy()
    merch_risk = merchant_risk[merchant_idx]
    logits = (
        -5.25
        + 0.62 * high_amount
        + 0.55 * far_distance
        + 0.85 * night
        + 0.72 * category_risk
        + 2.20 * ring_signal
        + 1.20 * card_risk
        + 1.35 * merch_risk
        + 1.05 * unusual.astype(float)
        + rng.normal(0, 0.45, n_transactions)
    )
    fraud_probability = _sigmoid(logits)
    is_fraud = rng.binomial(1, fraud_probability).astype(int)

    df = pd.DataFrame(
        {
            ID_COLUMN: [f"txn_{i:07d}" for i in range(n_transactions)],
            CARD_COLUMN: sampled_cards,
            "merchant": sampled_merchants,
            "gender": card_rows["gender"].to_numpy(),
            "job": card_rows["job"].to_numpy(),
            "category": sampled_categories,
            "city": card_rows["city"].to_numpy(),
            "state": card_rows["state"].to_numpy(),
            "zip": card_rows["zip"].to_numpy(),
            "trans_hour": hour.astype(str),
            "trans_minute": timestamps.minute.astype(str),
            "trans_second": timestamps.second.astype(str),
            "trans_year": timestamps.year.astype(str),
            "trans_month": timestamps.month.astype(str),
            "trans_day": timestamps.day.astype(str),
            "distance": distance.round(3),
            "age": card_rows["age"].to_numpy().astype(float),
            "amt": amount.round(2),
            "city_pop": card_rows["city_pop"].to_numpy().astype(float),
            TARGET: is_fraud,
        }
    )
    return df


def validate_dataset(df: pd.DataFrame, require_graph_columns: bool = True) -> None:
    required = set(CATEGORICAL_FEATURES + NUMERICAL_FEATURES + [TARGET, ID_COLUMN])
    if require_graph_columns:
        required.add(CARD_COLUMN)
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if df[TARGET].nunique() < 2:
        raise ValueError("Target column must contain both fraud and non-fraud examples")


def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, dtype={"zip": str})
    elif path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        raise ValueError("Supported dataset formats are CSV and Parquet")
    validate_dataset(df)
    return df


def split_dataset(
    df: pd.DataFrame,
    seed: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return deterministic 70/15/15 stratified train/validation/test frames."""
    train, temp = train_test_split(
        df,
        test_size=0.30,
        random_state=seed,
        stratify=df[TARGET],
    )
    val, test = train_test_split(
        temp,
        test_size=0.50,
        random_state=seed,
        stratify=temp[TARGET],
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
