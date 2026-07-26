# Course-to-Portfolio Mapping

This repository is an original engineering rewrite inspired by the concepts practised in an NVIDIA DLI anti-fraud course.

| Course concept | Portfolio implementation |
|---|---|
| Fraud metrics and thresholds | `src/fraud_detection/metrics.py` |
| CPU/GPU XGBoost and PR-AUC | `models/xgboost_model.py` |
| Categorical embeddings and MLP | `models/mlp.py` |
| Card–merchant graph construction | `models/gnn.py` (portable pure-PyTorch GraphSAGE rewrite) |
| GNN node embeddings for downstream XGBoost | `models/hybrid.py` |
| Triton FIL model repository | `scripts/export_triton.py` and `deployment/triton/` |

The original training notebooks are intentionally not redistributed. This keeps the public repository focused on the student's own implementation and avoids publishing course-provided material.
