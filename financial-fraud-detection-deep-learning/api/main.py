"""FastAPI prediction service."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from fraud_detection.inference import FraudPredictor


class TransactionRequest(BaseModel):
    distance: float = Field(12.5, ge=0)
    age: float = Field(36, ge=18, le=110)
    amt: float = Field(85.0, gt=0)
    city_pop: float = Field(100_000, ge=0)
    gender: str = "F"
    job: str = "job_01"
    category: str = "shopping_net"
    city: str = "city_01"
    state: str = "S01"
    zip: str = "10001"
    merchant: str = "merchant_001"
    trans_hour: str = "14"
    trans_minute: str = "30"
    trans_second: str = "00"
    trans_year: str = "2025"
    trans_month: str = "7"
    trans_day: str = "15"


@lru_cache(maxsize=1)
def get_predictor() -> FraudPredictor:
    model_dir = Path(os.getenv("MODEL_DIR", "artifacts/models"))
    return FraudPredictor(model_dir)


app = FastAPI(
    title="Financial Fraud Detection API",
    version="1.0.0",
    description="Portfolio-grade XGBoost inference API for transaction risk scoring.",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: TransactionRequest) -> dict[str, object]:
    try:
        return get_predictor().predict_one(payload.model_dump())
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
