# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Batch Inference
# MAGIC Scores the 750M-record inference universe against all production models.
# MAGIC
# MAGIC **Pipeline Stage**: Batch Inference (Distributed Scoring)
# MAGIC **Depends On**: Model Training (03)
# MAGIC
# MAGIC **Architecture**: Model-Sharded, Data-Parallel
# MAGIC - Models are iterated sequentially (or in small sub-batches)
# MAGIC - Each model's data is scored in parallel across Spark executors
# MAGIC - Model artifacts are broadcast to all executors

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

dbutils.widgets.text("catalog", "ml_platform", "Unity Catalog")
dbutils.widgets.text("schema", "production", "Schema")
dbutils.widgets.text("inference_date", "", "Inference Date")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
inference_date = dbutils.widgets.get("inference_date") or str(
    spark.sql("SELECT current_date()").collect()[0][0]
)

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")

print(f"Running inference for date: {inference_date}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Production Models from MLflow Registry

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Get all production models from the training results table
training_results = spark.table(f"{catalog}.{schema}.training_results").filter(
    "status = 'success'"
).toPandas()

production_models = []
for _, row in training_results.iterrows():
    model_id = row["model_id"]
    run_id = row["run_id"]
    if run_id:
        production_models.append({
            "model_id": model_id,
            "run_id": run_id,
            "model_uri": f"runs:/{run_id}/model",
        })

print(f"Found {len(production_models)} production models to score")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Inference Universe

# COMMAND ----------

# Load the inference universe (750M records in production)
# For demo, we use a smaller dataset
inference_table = f"{catalog}.{schema}.bronze_raw_data"
inference_df = spark.table(inference_table)

# Get feature columns (exclude metadata and targets)
exclude_cols = {"record_id", "target_binary", "target_continuous",
                "_ingestion_timestamp", "_run_date"}
feature_cols = [c for c in inference_df.columns if c not in exclude_cols
                and inference_df.schema[c].dataType.simpleString() in 
                ("double", "float", "int", "bigint", "long")]

print(f"Inference universe: {inference_df.count():,} records")
print(f"Feature columns: {len(feature_cols)}")

# Cache the inference data for repeated scoring
inference_df.cache()
inference_df.count()  # Materialize cache

# COMMAND ----------

# MAGIC %md
# MAGIC ## Distributed Scoring with mlflow.pyfunc.spark_udf

# COMMAND ----------

from pyspark.sql.functions import lit, current_timestamp, struct, col
from pyspark.sql.types import DoubleType
import time

results_table = f"{catalog}.{schema}.inference_results"
total_scored = 0
scoring_times = []

for idx, model_info in enumerate(production_models):
    model_id = model_info["model_id"]
    model_uri = model_info["model_uri"]
    
    print(f"\n[{idx+1}/{len(production_models)}] Scoring {model_id}...")
    start_time = time.time()
    
    try:
        # Create Spark UDF from MLflow model
        # This broadcasts the model to all executors automatically
        predict_udf = mlflow.pyfunc.spark_udf(
            spark, 
            model_uri=model_uri,
            result_type=DoubleType()
        )
        
        # Score the entire universe in parallel
        # Spark distributes this across all executors (150+ partitions × N executors)
        scored_df = (
            inference_df
            .withColumn("prediction", predict_udf(struct(*[col(c) for c in feature_cols])))
            .withColumn("model_id", lit(model_id))
            .withColumn("inference_date", lit(inference_date))
            .withColumn("scored_at", current_timestamp())
            .select("record_id", "model_id", "prediction", "inference_date", "scored_at")
        )
        
        # Write results (append, partitioned by model_id for efficient reads)
        scored_df.write.format("delta").mode("append").partitionBy(
            "model_id"
        ).saveAsTable(results_table)
        
        elapsed = time.time() - start_time
        num_scored = scored_df.count()
        throughput = num_scored / elapsed if elapsed > 0 else 0
        total_scored += num_scored
        scoring_times.append(elapsed)
        
        print(f"  ✅ {model_id}: {num_scored:,} records in {elapsed:.1f}s "
              f"({throughput:,.0f} rec/sec)")
              
    except Exception as e:
        print(f"  ❌ {model_id}: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference Summary

# COMMAND ----------

print(f"\n{'='*60}")
print(f"INFERENCE SUMMARY")
print(f"{'='*60}")
print(f"Total models scored: {len(scoring_times)}")
print(f"Total predictions: {total_scored:,}")
print(f"Average time per model: {sum(scoring_times)/len(scoring_times):.1f}s" if scoring_times else "N/A")
print(f"Total inference time: {sum(scoring_times)/60:.1f} minutes" if scoring_times else "N/A")

# Optimize results table
spark.sql(f"OPTIMIZE {results_table} ZORDER BY (model_id, record_id)")
print(f"\n✅ Results table optimized: {results_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validate Results

# COMMAND ----------

# Quick validation
results_check = spark.table(results_table)
print(f"Total result rows: {results_check.count():,}")
print(f"Distinct models: {results_check.select('model_id').distinct().count()}")
print(f"\nPrediction distribution:")
results_check.describe("prediction").show()

# Check for nulls
null_count = results_check.filter("prediction IS NULL").count()
print(f"Null predictions: {null_count:,}")
assert null_count == 0, f"Found {null_count} null predictions!"

print("\n✅ Inference validation passed")
