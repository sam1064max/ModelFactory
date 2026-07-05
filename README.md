# MLOps Pipeline — Small-Scale Implementation

A working demonstration of the MLOps platform architecture described in `architecture.md`, using recommended tools and orchestration patterns.

## Overview

This implementation demonstrates:
- **Data Ingestion** → Synthetic data generation with quality validation
- **Feature Engineering** → PySpark-style transforms with feature selection
- **Model Training** → XGBoost/LightGBM with MLflow tracking
- **Batch Inference** → Parallel scoring with chunked processing
- **Monitoring** → Drift detection with Evidently AI
- **Orchestration** → Local pipeline runner + Databricks Workflow configs
- **Infrastructure** → Terraform templates for Databricks provisioning

## Quick Start (Local Demo)

### Prerequisites
- Python 3.10+
- pip

### Setup

```bash
cd mlops_pipeline
pip install -r requirements.txt
```

### Run the Full Pipeline

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
