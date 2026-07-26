"""Streamlit dashboard for interactive transaction risk scoring."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from fraud_detection.inference import FraudPredictor

st.set_page_config(page_title="Fraud Detection Demo", page_icon="🛡️", layout="wide")
st.title("Financial Fraud Detection")
st.caption("Interactive demo powered by the trained XGBoost baseline. Synthetic data only.")


@st.cache_resource
def load_predictor() -> FraudPredictor:
    return FraudPredictor(Path(os.getenv("MODEL_DIR", "artifacts/models")))


try:
    predictor = load_predictor()
except FileNotFoundError:
    st.error("Model artifacts are missing. Run `python scripts/train_all.py` first.")
    st.stop()

left, right = st.columns(2)
with left:
    st.subheader("Transaction")
    amt = st.number_input("Amount", min_value=0.01, value=85.0, step=10.0)
    distance = st.number_input("Distance from home (km)", min_value=0.0, value=12.5)
    category = st.selectbox(
        "Category",
        ["shopping_net", "grocery_pos", "travel", "misc_net", "food_dining"],
    )
    merchant = st.text_input("Merchant ID", "merchant_001")
    trans_hour = str(st.slider("Transaction hour", 0, 23, 14))

with right:
    st.subheader("Customer context")
    age = st.slider("Age", 18, 90, 36)
    gender = st.selectbox("Gender", ["F", "M"])
    job = st.text_input("Job category", "job_01")
    city = st.text_input("City", "city_01")
    state = st.text_input("State", "S01")
    zip_code = st.text_input("ZIP", "10001")
    city_pop = st.number_input("City population", min_value=0.0, value=100_000.0)

transaction = {
    "distance": distance,
    "age": age,
    "amt": amt,
    "city_pop": city_pop,
    "gender": gender,
    "job": job,
    "category": category,
    "city": city,
    "state": state,
    "zip": zip_code,
    "merchant": merchant,
    "trans_hour": trans_hour,
    "trans_minute": "30",
    "trans_second": "00",
    "trans_year": "2025",
    "trans_month": "7",
    "trans_day": "15",
}

if st.button("Score transaction", type="primary", use_container_width=True):
    result = predictor.predict_one(transaction)
    probability = float(result["fraud_probability"])
    st.metric("Fraud probability", f"{probability:.2%}")
    st.progress(min(max(probability, 0.0), 1.0))
    if result["risk_level"] == "high":
        st.error("High risk — block or require step-up authentication.")
    elif result["risk_level"] == "medium":
        st.warning("Medium risk — request additional verification.")
    else:
        st.success("Low risk — approve under the current policy.")
    st.json(result)

with st.expander("Input payload"):
    st.dataframe(pd.DataFrame([transaction]), use_container_width=True)
