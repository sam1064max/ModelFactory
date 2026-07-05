# MLOps Pipeline — Small-Scale Implementation

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%20|%203.11%20|%203.12-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
  <img src="https://img.shields.io/github/actions/workflow/status/sam1064max/ModelFactory/ci.yml?branch=main&label=CI%2FCD&logo=github" alt="CI/CD">
  <img src="https://img.shields.io/github/last-commit/sam1064max/ModelFactory" alt="Last Commit">
  <img src="https://img.shields.io/github/v/release/sam1064max/ModelFactory" alt="Release">
  <img src="https://img.shields.io/pypi/v/mlops-platform-ray?logo=pypi&logoColor=white" alt="PyPI">
  <img src="https://img.shields.io/badge/Ray-2.9%2B-028CF0?logo=ray&logoColor=white" alt="Ray">
  <img src="https://img.shields.io/badge/MLflow-2.10%2B-0194E2?logo=mlflow&logoColor=white" alt="MLflow">
  <img src="https://img.shields.io/badge/Terraform-1.7%2B-7B42BC?logo=terraform&logoColor=white" alt="Terraform">
  <img src="https://img.shields.io/badge/Streamlit-1.36%2B-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
</p>

A working demonstration of the MLOps platform architecture described in [`architecture_final.md`](architecture_final.md), using recommended tools and orchestration patterns. The full architecture document covers the production-scale design for **10,000 models**, **7.5 trillion daily predictions**, and **< 24-hour inference SLA** on Databricks Lakehouse with Spark + Ray.

## Overview

This implementation demonstrates:
- **Data Ingestion** → Synthetic data generation with quality validation
- **Feature Engineering** → PySpark-style transforms with feature selection
- **Model Training** → XGBoost/LightGBM with MLflow tracking
- **Batch Inference** → Parallel scoring with chunked processing
- **Monitoring** → Drift detection with Evidently AI
- **Orchestration** → Local pipeline runner + Databricks Workflow configs
- **Infrastructure** → Terraform templates for Databricks provisioning
- **Frontend** → Interactive Streamlit dashboard with live pipeline visualization

## Quick Start (Local Demo)

### Prerequisites
- Python 3.10+
- pip

### Setup

```bash
cd mlops_pipeline
pip install -r requirements.txt
```

### Run the Full Pipeline (CLI)

```bash
python -m orchestration.pipeline_runner
```

This will:
1. Generate synthetic training data (10K rows, 50 features)
2. Run feature engineering (50 → ~200 transformed features)
3. Select top features (200 → 30 features per model)
4. Train 10 models (5 classification + 5 regression) with MLflow tracking
5. Run batch inference on synthetic universe (100K rows)
6. Run drift detection (Evidently reports)

### Launch the Streamlit Dashboard

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` for an interactive pipeline dashboard:

- **Pipeline Execution** — Run the full pipeline from the UI with configurable parameters
- **Live Logs** — Real-time log streaming as each stage executes
- **Metrics Dashboard** — KPI cards for trained models, features, drift, and timing
- **Architecture Page** — Graphviz diagram with full system overview

> ![Pipeline Dashboard](docs/screenshots/pipeline.png)
> ![Architecture Page](docs/screenshots/architecture.png)
> *(Add screenshots to `docs/screenshots/` after running locally)*

### View MLflow Dashboard

```bash
mlflow ui --port 5000
```

Navigate to `http://localhost:5000` to view experiments, metrics, and registered models.

## Databricks Deployment

### Deploy with Databricks Asset Bundles

```bash
# Configure Databricks CLI
databricks configure --token

# Validate the bundle
databricks bundle validate -t staging

# Deploy to staging
databricks bundle deploy -t staging

# Deploy to production
databricks bundle deploy -t prod
```

### Deploy Infrastructure with Terraform

```bash
cd infrastructure/terraform
terraform init
terraform plan -var-file="prod.tfvars"
terraform apply -var-file="prod.tfvars"
```

## Project Structure

```
mlops_pipeline/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── app.py                             # Streamlit dashboard (main entry)
├── pages/
│   └── Architecture.py                # Architecture diagram page
├── frontend/
│   └── pipeline_runner.py             # Streamlit-aware pipeline wrapper
├── .streamlit/
│   └── config.toml                    # Streamlit theme & server config
├── config/
│   ├── pipeline_config.yaml           # Pipeline configuration
│   └── model_registry.yaml            # Model definitions (10 demo models)
├── src/
│   ├── __init__.py
│   ├── data_ingestion.py              # Synthetic data generation + validation
│   ├── feature_engineering.py         # Feature transformation pipeline
│   ├── model_training.py              # Training with MLflow tracking
│   ├── batch_inference.py             # Chunked batch inference engine
│   ├── model_monitoring.py            # Drift detection & alerting
│   └── utils.py                       # Shared utilities & logging
├── orchestration/
│   ├── pipeline_runner.py             # Local end-to-end orchestration
│   └── databricks_workflow.json       # Databricks Workflows job config
├── databricks/
│   ├── notebooks/
│   │   ├── 01_data_ingestion.py       # Databricks notebook: ingestion
│   │   ├── 02_feature_engineering.py  # Databricks notebook: features
│   │   ├── 03_model_training.py       # Databricks notebook: training
│   │   ├── 04_batch_inference.py      # Databricks notebook: inference
│   │   └── 05_monitoring.py           # Databricks notebook: monitoring
│   └── databricks.yml                 # Databricks Asset Bundle config
├── infrastructure/
│   └── terraform/
│       ├── main.tf                    # Databricks workspace + clusters
│       ├── variables.tf               # Configurable variables
│       └── outputs.tf                 # Terraform outputs
├── monitoring/
│   └── drift_report_config.json       # Evidently report configuration
└── tests/
    ├── test_feature_engineering.py     # Feature transform unit tests
    └── test_model_training.py         # Training pipeline unit tests
```

## Key Design Patterns Demonstrated

| Pattern | Implementation |
|---|---|
| **Config-Driven Models** | `model_registry.yaml` defines all model params — no code changes for new models |
| **MLflow Integration** | Every training run logs params, metrics, artifacts, and registers models |
| **Champion/Challenger** | New models compared against existing production models before promotion |
| **Chunked Inference** | Large datasets processed in configurable chunks for memory efficiency |
| **Drift Detection** | Evidently AI computes PSI, KS-test, and data quality metrics |
| **Reproducibility** | Random seeds, data versioning (hashes), and environment pinning |
