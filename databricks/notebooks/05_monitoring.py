# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Monitoring & Drift Detection
# MAGIC Monitors model and data drift using Databricks Lakehouse Monitoring patterns.
# MAGIC
# MAGIC **Pipeline Stage**: Post-Inference Monitoring
# MAGIC **Depends On**: Batch Inference (04)

# COMMAND ----------

dbutils.widgets.text("catalog", "ml_platform", "Unity Catalog")
dbutils.widgets.text("schema", "production", "Schema")
dbutils.widgets.text("run_date", "", "Run Date")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
run_date = dbutils.widgets.get("run_date") or str(
    spark.sql("SELECT current_date()").collect()[0][0]
)

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Prediction Distribution Monitoring

# COMMAND ----------

from pyspark.sql.functions import (
    col, count, mean, stddev, min as spark_min, max as spark_max,
    percentile_approx, lit, current_timestamp
)

results_table = f"{catalog}.{schema}.inference_results"
results_df = spark.table(results_table)

# Compute prediction statistics per model
pred_stats = (
    results_df
    .groupBy("model_id")
    .agg(
        count("*").alias("record_count"),
        mean("prediction").alias("pred_mean"),
        stddev("prediction").alias("pred_std"),
        spark_min("prediction").alias("pred_min"),
        spark_max("prediction").alias("pred_max"),
        percentile_approx("prediction", 0.5).alias("pred_median"),
        percentile_approx("prediction", 0.1).alias("pred_p10"),
        percentile_approx("prediction", 0.9).alias("pred_p90"),
    )
    .withColumn("run_date", lit(run_date))
    .withColumn("computed_at", current_timestamp())
)

# Save prediction stats
stats_table = f"{catalog}.{schema}.prediction_stats"
pred_stats.write.format("delta").mode("append").saveAsTable(stats_table)

print(f"✅ Prediction stats saved for {pred_stats.count()} models")
pred_stats.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Feature Drift Detection (PSI)

# COMMAND ----------

from pyspark.sql.functions import (
    expr, log as spark_log, sum as spark_sum,
    ntile, dense_rank
)
from pyspark.sql.window import Window

# Compare training distribution (Gold features) vs inference features
gold_table = f"{catalog}.{schema}.gold_features"
gold_df = spark.table(gold_table)

# Get numeric feature columns
numeric_features = [c for c in gold_df.columns 
                    if c != "record_id" 
                    and gold_df.schema[c].dataType.simpleString() in ("double", "float")]

# PSI computation for each feature using SQL window functions
# This is the scalable approach — one pass over the data
PSI_THRESHOLD = 0.2
drift_results = []

for feature in numeric_features[:30]:  # Limit for demo
    # Compute decile bins from training data
    train_quantiles = gold_df.approxQuantile(feature, 
        [i/10 for i in range(11)], 0.01)
    
    if len(set(train_quantiles)) < 3:
        continue  # Skip constant/near-constant features
    
    bins = sorted(set(train_quantiles))
    
    # Bucketize both distributions
    from pyspark.ml.feature import Bucketizer
    
    splits = [float("-inf")] + bins[1:-1] + [float("inf")]
    splits = sorted(set(splits))
    
    if len(splits) < 3:
        continue
    
    bucketizer = Bucketizer(
        splits=splits, inputCol=feature, outputCol="bucket",
        handleInvalid="keep"
    )
    
    try:
        train_buckets = (
            bucketizer.transform(gold_df.select(feature).na.fill(0))
            .groupBy("bucket").count()
            .withColumn("pct", col("count") / gold_df.count())
        ).toPandas()
        
        inference_buckets = (
            bucketizer.transform(
                results_df.select(col("prediction").alias(feature)).na.fill(0)
            )
            .groupBy("bucket").count()
            .withColumn("pct", col("count") / results_df.count())
        ).toPandas()
        
        # Calculate PSI
        import numpy as np
        merged = train_buckets.merge(inference_buckets, on="bucket", 
                                      suffixes=("_train", "_inf"), how="outer").fillna(0.001)
        
        psi = np.sum(
            (merged["pct_inf"] - merged["pct_train"]) * 
            np.log(merged["pct_inf"] / merged["pct_train"])
        )
        
        drift_results.append({
            "feature": feature,
            "psi": float(psi),
            "is_drifted": psi > PSI_THRESHOLD,
            "run_date": run_date,
        })
    except Exception as e:
        print(f"  Skipping {feature}: {e}")

# Save drift results
if drift_results:
    drift_df = spark.createDataFrame(drift_results)
    drift_table = f"{catalog}.{schema}.feature_drift_metrics"
    drift_df.write.format("delta").mode("append").saveAsTable(drift_table)
    
    drifted_count = sum(1 for r in drift_results if r["is_drifted"])
    print(f"\n✅ Drift detection complete:")
    print(f"   Features analyzed: {len(drift_results)}")
    print(f"   Features drifted: {drifted_count}")
    drift_df.filter("is_drifted = true").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Alerting

# COMMAND ----------

import json
import requests

# Check alert conditions
alerts = []

# P1: Major drift
if drift_results:
    drift_pct = sum(1 for r in drift_results if r["is_drifted"]) / len(drift_results)
    if drift_pct > 0.5:
        alerts.append({
            "severity": "P1",
            "message": f"{drift_pct:.0%} of features show significant drift",
            "action": "Trigger emergency retraining"
        })
    elif drift_pct > 0.2:
        alerts.append({
            "severity": "P2", 
            "message": f"{drift_pct:.0%} of features show drift",
            "action": "Schedule retraining for affected models"
        })

# P2: Prediction anomalies
pred_stats_pd = pred_stats.toPandas()
for _, row in pred_stats_pd.iterrows():
    if row["pred_std"] < 1e-10:
        alerts.append({
            "severity": "P1",
            "message": f"Model {row['model_id']} producing constant predictions",
            "action": "Investigate model artifact"
        })

# Send alerts (mock for demo)
if alerts:
    print(f"\n⚠️ {len(alerts)} ALERTS GENERATED:")
    for alert in alerts:
        print(f"  [{alert['severity']}] {alert['message']}")
        print(f"       Action: {alert['action']}")
    
    # In production: send to PagerDuty/Slack
    # slack_webhook = dbutils.secrets.get(scope="ml-platform", key="slack-webhook")
    # requests.post(slack_webhook, json={"text": json.dumps(alerts, indent=2)})
else:
    print("\n✅ No alerts — all monitors within thresholds")

# Save alert log
alert_table = f"{catalog}.{schema}.monitoring_alerts"
if alerts:
    alert_df = spark.createDataFrame(alerts).withColumn(
        "run_date", lit(run_date)
    ).withColumn("created_at", current_timestamp())
    alert_df.write.format("delta").mode("append").saveAsTable(alert_table)
