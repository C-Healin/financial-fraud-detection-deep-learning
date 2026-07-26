"""Minimal HTTP client for the exported Triton FIL model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tritonclient.http as httpclient

from fraud_detection.preprocessing import FraudPreprocessor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="localhost:8000")
    parser.add_argument("--model-name", default="fraud_xgboost")
    parser.add_argument("--sample", type=Path, default=Path("data/sample/fraud_sample.csv"))
    parser.add_argument("--preprocessor", type=Path, default=Path("artifacts/models/xgboost_preprocessor.joblib"))
    args = parser.parse_args()

    frame = pd.read_csv(args.sample, dtype={"zip": str}).iloc[[0]]
    preprocessor = FraudPreprocessor.load(args.preprocessor)
    matrix = preprocessor.transform_tabular(frame).astype(np.float32)

    client = httpclient.InferenceServerClient(url=args.url)
    triton_input = httpclient.InferInput("input__0", matrix.shape, "FP32")
    triton_input.set_data_from_numpy(matrix)
    output = httpclient.InferRequestedOutput("output__0")
    response = client.infer(args.model_name, inputs=[triton_input], outputs=[output])
    print(response.as_numpy("output__0"))


if __name__ == "__main__":
    main()
