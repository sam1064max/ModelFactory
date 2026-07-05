# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Model Training
# MAGIC Trains a batch of models with Hyperopt HPO and MLflow tracking.
# MAGIC
# MAGIC **Pipeline Stage**: Model Training (Parallel Batch)
# MAGIC **Depends On**: Feature Engineering (02)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

dbutils.widgets.text("catalog", "ml_platform", "Unity Catalog")
dbutils.widgets.text("schema", "production", "Schema")
dbutils.widgets.text("model_batch_start", "0", "Model Batch Start Index")
dbutils.widgets.text("model_batch_end", "2500", "Model Batch End Index")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
batch_start = int(dbutils.widgets.get("model_batch_start"))
batch_end = int(dbutils.widgets.get("model_batch_end"))

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")

print(f"Training models {batch_start} to {batch_end}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Model Configurations

# COMMAND ----------

import yaml
import os

# Load model registry (in production, this would be from a Delta table or Unity Catalog volume)
# For demo, we inline a subset of configs
model_configs = [
    {
        "model_id": f"model_{i:05d}",
        "model_type": "xgboost_classifier",
        "target_column": "target_binary",
        "feature_table": f"{catalog}.{schema}.gold_features",
        "hyperparameter_space": {
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "n_estimators": [100, 300],
            "subsample": [0.8, 0.9],
        },
    }
    for i in range(batch_start, batch_end)
]

print(f"Loaded {len(model_configs)} model configurations")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training Function with Hyperopt + MLflow

# COMMAND ----------

import mlflow
import mlflow.xgboost
import xgboost as xgb
import numpy as np
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hyperopt import SparkTrials
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
import pandas as pd


def train_single_model(model_config, feature_df_pd, max_evals=10):
    """
    Train a single model with Hyperopt Bayesian optimization.
    
    In production, feature_df_pd would be loaded from the Feature Store
    using `fs.read_table()` with point-in-time lookup.
    """
    model_id = model_config["model_id"]
    target_col = model_config["target_column"]
    
    # Prepare data
    feature_cols = [c for c in feature_df_pd.columns 
                    if c not in [target_col, "record_id", "target_continuous",
                                "_ingestion_timestamp", "_run_date"]
                    and feature_df_pd[c].dtype in ['float64', 'int64', 'float32', 'int32']]
    
    X = feature_df_pd[feature_cols].fillna(0).values
    y = feature_df_pd[target_col].values
    
    # Define Hyperopt search space
    space = {
        "max_depth": hp.choice("max_depth", model_config["hyperparameter_space"]["max_depth"]),
        "learning_rate": hp.choice("learning_rate", model_config["hyperparameter_space"]["learning_rate"]),
        "n_estimators": hp.choice("n_estimators", model_config["hyperparameter_space"]["n_estimators"]),
        "subsample": hp.choice("subsample", model_config["hyperparameter_space"]["subsample"]),
    }
    
    def objective(params):
        with mlflow.start_run(nested=True):
            model = xgb.XGBClassifier(
                **params,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
                verbosity=0,
            )
            
            # Cross-validation
            scores = cross_val_score(model, X, y, cv=3, scoring="roc_auc")
            mean_auc = scores.mean()
            
            # Log to MLflow
            mlflow.log_params(params)
            mlflow.log_metric("cv_auc_mean", mean_auc)
            mlflow.log_metric("cv_auc_std", scores.std())
            
            return {"loss": -mean_auc, "status": STATUS_OK}
    
    # Run Hyperopt with SparkTrials for distributed HPO
    with mlflow.start_run(run_name=model_id) as parent_run:
        mlflow.set_tags({
            "model_id": model_id,
            "model_type": model_config["model_type"],
            "batch_range": f"{batch_start}-{batch_end}",
        })
        
        # Use SparkTrials for distributed Bayesian optimization
        spark_trials = SparkTrials(parallelism=4)
        
        best_params_idx = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=spark_trials,
        )
        
        # Map indices back to actual values
        hp_space = model_config["hyperparameter_space"]
        best_params = {
            "max_depth": hp_space["max_depth"][best_params_idx["max_depth"]],
            "learning_rate": hp_space["learning_rate"][best_params_idx["learning_rate"]],
            "n_estimators": hp_space["n_estimators"][best_params_idx["n_estimators"]],
            "subsample": hp_space["subsample"][best_params_idx["subsample"]],
        }
        
        # Train final model on all data with best params
        final_model = xgb.XGBClassifier(
            **best_params,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
        final_model.fit(X, y)
        
        # Evaluate
        y_prob = final_model.predict_proba(X)[:, 1]
        train_auc = roc_auc_score(y, y_prob)
        
        # Log final model
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("final_train_auc", train_auc)
        mlflow.xgboost.log_model(
            final_model,
            artifact_path="model",
            registered_model_name=model_id,
        )
        
        print(f"  ✅ {model_id}: AUC={train_auc:.4f}, params={best_params}")
        
        return {
            "model_id": model_id,
            "run_id": parent_run.info.run_id,
            "best_params": best_params,
            "train_auc": train_auc,
        }

# COMMAND ----------

# MAGIC %md
# MAGIC ## Execute Training Batch

# COMMAND ----------

# Load feature data (sample for demo; in production, use Feature Store)
feature_table = f"{catalog}.{schema}.gold_features"

# Check if feature table exists, otherwise use bronze
try:
    feature_df = spark.table(feature_table).toPandas()
    print(f"Loaded features from {feature_table}: {feature_df.shape}")
except Exception:
    print(f"Feature table {feature_table} not found, using bronze data")
    feature_df = spark.table(f"{catalog}.{schema}.bronze_raw_data").limit(50000).toPandas()
    print(f"Loaded bronze data: {feature_df.shape}")

# COMMAND ----------

# Set MLflow experiment
experiment_name = f"/ml-platform/training-batch-{batch_start}-{batch_end}"
mlflow.set_experiment(experiment_name)

# Train models in batch
results = []
for i, config in enumerate(model_configs[:5]):  # Limit for demo
    print(f"\n[{i+1}/{min(len(model_configs), 5)}] Training {config['model_id']}...")
    try:
        result = train_single_model(config, feature_df, max_evals=5)
        results.append(result)
    except Exception as e:
        print(f"  ❌ {config['model_id']}: {e}")
        results.append({"model_id": config["model_id"], "error": str(e)})

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Training Results

# COMMAND ----------

results_df = spark.createDataFrame(
    pd.DataFrame([
        {
            "model_id": r.get("model_id"),
            "run_id": r.get("run_id", ""),
            "train_auc": r.get("train_auc", 0.0),
            "status": "success" if "run_id" in r else "failed",
            "error": r.get("error", ""),
        }
        for r in results
    ])
)

results_table = f"{catalog}.{schema}.training_results"
results_df.write.format("delta").mode("append").saveAsTable(results_table)
print(f"\n✅ Training results saved to {results_table}")
print(f"   Successful: {sum(1 for r in results if 'run_id' in r)}")
print(f"   Failed: {sum(1 for r in results if 'error' in r)}")
