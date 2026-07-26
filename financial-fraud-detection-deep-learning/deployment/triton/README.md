# NVIDIA Triton FIL deployment

This directory is generated from the trained XGBoost model.

```bash
python scripts/export_triton.py
```

The export creates:

```text
model_repository/
└── fraud_xgboost/
    ├── 1/
    │   └── xgboost.json
    └── config.pbtxt
```

Run Triton with an NVIDIA GPU and a compatible Triton container:

```bash
docker run --rm --gpus all \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v "$PWD/deployment/triton/model_repository:/models" \
  nvcr.io/nvidia/tritonserver:<release>-py3 \
  tritonserver --model-repository=/models
```

Use a Triton release compatible with the XGBoost serialization version used to train the model. The FastAPI and Streamlit demos do not require Triton.
