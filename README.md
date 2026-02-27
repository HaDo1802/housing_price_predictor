[![ml-pipeline-ci](https://github.com/HaDo1802/housing_price_predictor/actions/workflows/ml_pipeline_ci.yml/badge.svg)](https://github.com/HaDo1802/housing_price_predictor/actions/workflows/ml_pipeline_ci.yml)

# Housing Price Predictor

Production-oriented ML project for house-price prediction using the Ames Housing dataset.

## Live App
https://huggingface.co/spaces/HaDo1802/housing-predictor

## What This Repo Demonstrates
- Data-leakage-safe train/val/test workflow
- Reproducible preprocessing + model artifacts
- MLflow experiment tracking and registry flow
- Separation of ML core code, serving code, and operational entrypoints

## Project Structure

```text
housing_price_predictor/
├── conf/                            # Config system (base + environment overrides)
│   ├── base/
│   │   ├── data.yaml
│   │   ├── model.yaml
│   │   ├── preprocessing.yaml
│   │   └── training.yaml
│   ├── local/
│   │   └── data.yaml
│   ├── production/
│   │   └── data.yaml
│   ├── config.yaml                  # Main editable config
│   └── config_manager.py
│
├── src/
│   └── housing_predictor/           # Installable ML package
│       ├── data/                    # loader / cleaner / splitter
│       ├── features/                # preprocessor / schema
│       ├── models/                  # trainer / evaluator / registry
│       ├── pipelines/               # orchestration (training/inference)
│       └── monitoring/              # feedback + drift utilities
│
├── serving/
│   ├── api/                         # FastAPI app + routers + schemas
│   │   ├── main.py
│   │   ├── schemas.py
│   │   └── routers/
│   ├── app/
│   │   └── streamlit_app.py         # Streamlit UI
│   └── vercel/
│       └── index.py                 # Vercel entrypoint
│
├── pipelines/                       # Executable job entrypoints
│   ├── run_training.py
│   ├── run_tuning.py
│   ├── run_promote.py
│   └── run_feedback_monitor.py
│
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.streamlit
│   └── docker-compose.yml
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── conftest.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── feedback/
│
├── models/
│   ├── production/
│   └── experiments/
│
├── notebooks/
├── docs/
├── pyproject.toml
├── requirements.txt
├── Makefile
└── vercel.json
```

## Core Workflow

```text
Raw data -> split -> preprocess (fit on train only) -> train -> evaluate -> register/promote -> serve
```

## Running Locally

### 1) Install
```bash
python -m pip install -r requirements.txt
```

### 2) Run API
```bash
make api
# or: python -m uvicorn serving.api.main:app --host 0.0.0.0 --port 8000
```

### 3) Run Streamlit UI
```bash
make ui
# or: python -m streamlit run serving/app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

### 4) Run training pipeline
```bash
python pipelines/run_training.py
```

### 5) Run tuning / promote / monitor
```bash
python pipelines/run_tuning.py
python pipelines/run_promote.py --list-only
python pipelines/run_feedback_monitor.py
```

## Docker

```bash
docker compose -f docker/docker-compose.yml up --build
```

## MLflow

```bash
mlflow ui
```
Open: `http://localhost:5000`

## Artifacts
- `models/production/model.pkl`
- `models/production/preprocessor.pkl`
- `models/production/config.yaml`
- `models/production/metadata.json`

## Notes
- Use `conf/config.yaml` for main config edits.
- Environment-specific overrides are in `conf/local/` and `conf/production/`.
- API routes are organized by concern under `serving/api/routers/`.

## Author
Ha Do  
GitHub: https://github.com/HaDo1802
