"""Shared feature definitions used by training and inference."""

CATEGORICAL_FEATURES = [
    "gender",
    "job",
    "category",
    "city",
    "state",
    "zip",
    "merchant",
    "trans_hour",
    "trans_minute",
    "trans_second",
    "trans_year",
    "trans_month",
    "trans_day",
]

NUMERICAL_FEATURES = ["distance", "age", "amt", "city_pop"]
TABULAR_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET = "is_fraud"
ID_COLUMN = "transaction_id"
CARD_COLUMN = "card"
MERCHANT_COLUMN = "merchant"

RANDOM_STATE = 42
