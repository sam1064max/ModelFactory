# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Feature Engineering
# MAGIC Transforms raw Bronze data into engineered features in the Gold layer.
# MAGIC Registers features in the Databricks Feature Store (Unity Catalog).
# MAGIC
# MAGIC **Pipeline Stage**: Silver → Gold (Feature Tables)
# MAGIC **Depends On**: Data Ingestion (01)

# COMMAND ----------

dbutils.widgets.text("catalog", "ml_platform", "Unity Catalog")
dbutils.widgets.text("schema", "production", "Schema")
dbutils.widgets.text("feature_set", "fs_all_features", "Feature Set ID")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
feature_set = dbutils.widgets.get("feature_set")

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Bronze Data

# COMMAND ----------

from pyspark.sql.functions import (
    col, when, log1p, abs as spark_abs, pow as spark_pow,
    mean as spark_mean, stddev, isnull, lit,
    dayofweek, month, dayofyear, sin as spark_sin, cos as spark_cos
)
from pyspark.ml.feature import (
    StandardScaler, VectorAssembler, StringIndexer,
    OneHotEncoder, SQLTransformer
)
from pyspark.ml import Pipeline
import math

bronze_table = f"{catalog}.{schema}.bronze_raw_data"
bronze_df = spark.table(bronze_table)

print(f"Bronze data: {bronze_df.count():,} rows × {len(bronze_df.columns)} columns")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Phase 1: Broad Feature Transformation (PySpark)

# COMMAND ----------

# Identify column types
numeric_cols = [c for c in bronze_df.columns if c.startswith("num_")]
cat_cols = [c for c in bronze_df.columns if c.startswith("cat_")]
date_cols = [c for c in bronze_df.columns if c.startswith("date_")]

print(f"Numeric: {len(numeric_cols)}, Categorical: {len(cat_cols)}, Temporal: {len(date_cols)}")

# ── Numeric Transforms ───────────────────────────────────────────────────
features_df = bronze_df

for col_name in numeric_cols:
    # Null indicator
    features_df = features_df.withColumn(
        f"{col_name}_is_null", isnull(col(col_name)).cast("double")
    )
    # Fill nulls with 0 for transforms
    features_df = features_df.withColumn(
        f"{col_name}_filled", when(isnull(col(col_name)), 0.0).otherwise(col(col_name))
    )
    # Log transform
    features_df = features_df.withColumn(
        f"{col_name}_log1p", log1p(spark_abs(col(f"{col_name}_filled")))
    )
    # Squared
    features_df = features_df.withColumn(
        f"{col_name}_squared", spark_pow(col(f"{col_name}_filled"), 2)
    )

# ── Categorical Transforms (StringIndexer) ──────────────────────────────
indexers = [
    StringIndexer(inputCol=c, outputCol=f"{c}_indexed", handleInvalid="keep")
    for c in cat_cols
]

pipeline = Pipeline(stages=indexers)
features_df = pipeline.fit(features_df).transform(features_df)

# ── Top Feature Interactions ─────────────────────────────────────────────
top_numeric = numeric_cols[:5]
for i in range(len(top_numeric)):
    for j in range(i + 1, len(top_numeric)):
        a, b = top_numeric[i], top_numeric[j]
        features_df = features_df.withColumn(
            f"interact_{a}_{b}",
            col(f"{a}_filled") * col(f"{b}_filled")
        )

print(f"After transforms: {len(features_df.columns)} columns")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register in Feature Store (Unity Catalog)

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

# Select feature columns (exclude raw, metadata, and targets)
exclude = {"record_id", "target_binary", "target_continuous",
           "_ingestion_timestamp", "_run_date"}
exclude.update(set(numeric_cols))  # Exclude raw numeric (keep transformed)
exclude.update(set(cat_cols))      # Exclude raw categorical (keep indexed)

feature_columns = [c for c in features_df.columns if c not in exclude]
gold_df = features_df.select("record_id", *feature_columns)

# Write to Gold feature table
gold_table = f"{catalog}.{schema}.gold_features"
gold_df.write.format("delta").mode("overwrite").saveAsTable(gold_table)

# Register as Feature Table in Unity Catalog
try:
    fe.create_table(
        name=gold_table,
        primary_keys=["record_id"],
        description=f"Engineered features for {feature_set}. "
                    f"{len(feature_columns)} features from {len(numeric_cols)} numeric, "
                    f"{len(cat_cols)} categorical source columns.",
        df=gold_df,
    )
    print(f"✅ Feature table registered: {gold_table}")
except Exception as e:
    # Table may already exist
    print(f"Feature table update: {e}")
    gold_df.write.format("delta").mode("overwrite").option(
        "overwriteSchema", "true"
    ).saveAsTable(gold_table)
    print(f"✅ Feature table overwritten: {gold_table}")

print(f"   Features: {len(feature_columns)}")
print(f"   Rows: {gold_df.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Optimize Gold Table

# COMMAND ----------

spark.sql(f"OPTIMIZE {gold_table} ZORDER BY (record_id)")
print(f"✅ Gold feature table optimized")
