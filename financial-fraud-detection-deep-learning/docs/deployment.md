# Deployment Guide

## Local FastAPI

```bash
uvicorn api.main:app --reload
curl http://localhost:8000/health
```

Open `http://localhost:8000/docs` for the generated OpenAPI interface.

## Local Streamlit

```bash
streamlit run app/streamlit_app.py
```

## Docker Compose

```bash
docker compose up --build
```

- API: `http://localhost:8000/docs`
- Streamlit: `http://localhost:8501`

## Production considerations

Add authentication, request logging with PII redaction, a feature store, schema validation, monitoring, drift detection, rollback support, and a human-review workflow before production use.
