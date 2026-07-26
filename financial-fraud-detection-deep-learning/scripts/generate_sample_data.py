#!/usr/bin/env python3
"""Generate a publishable synthetic fraud dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from fraud_detection.data import generate_synthetic_fraud_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=12_000)
    parser.add_argument("--output", type=Path, default=Path("data/sample/fraud_sample.csv"))
    args = parser.parse_args()
    df = generate_synthetic_fraud_data(n_transactions=args.rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df):,} rows to {args.output}")
    print(f"Fraud rate: {df['is_fraud'].mean():.2%}")


if __name__ == "__main__":
    main()
