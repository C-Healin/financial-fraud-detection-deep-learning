#!/usr/bin/env python3
"""Export the trained XGBoost model to an NVIDIA Triton FIL repository."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/models"))
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path("deployment/triton/model_repository/fraud_xgboost"),
    )
    parser.add_argument("--max-batch-size", type=int, default=32768)
    args = parser.parse_args()

    source_model = args.artifact_dir / "xgboost_model.json"
    metadata = json.loads((args.artifact_dir / "xgboost_metadata.json").read_text())
    if not source_model.exists():
        raise FileNotFoundError("Train the XGBoost model before exporting to Triton")

    version_dir = args.repository / "1"
    version_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_model, version_dir / "model.json")
    n_features = len(metadata["feature_names"])
    config = f'''backend: "fil"
max_batch_size: {args.max_batch_size}
input [
  {{
    name: "input__0"
    data_type: TYPE_FP32
    dims: [ {n_features} ]
  }}
]
output [
  {{
    name: "output__0"
    data_type: TYPE_FP32
    dims: [ 1 ]
  }}
]
instance_group [{{ kind: KIND_AUTO }}]
parameters [
  {{ key: "model_type" value: {{ string_value: "xgboost_json" }} }},
  {{ key: "is_classifier" value: {{ string_value: "true" }} }}
]
dynamic_batching {{}}
'''
    (args.repository / "config.pbtxt").write_text(config, encoding="utf-8")
    print(f"Triton repository exported to {args.repository}")


if __name__ == "__main__":
    main()
