#!/usr/bin/env python3
"""Train XGBoost, embedding MLP, GraphSAGE, and hybrid models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fraud_detection.pipeline import run_training_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/sample/fraud_sample.csv"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()
    metrics = run_training_pipeline(args.data, args.project_root)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
