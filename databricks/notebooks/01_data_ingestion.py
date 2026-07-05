# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Data Ingestion
# MAGIC Ingests raw data into the Bronze layer using Auto Loader and Delta Live Tables patterns.
# MAGIC
# MAGIC **Pipeline Stage**: Bronze Layer (Raw Data)
# MAGIC **Trigger**: Scheduled daily or on data arrival

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

dbutils.widgets.text("catalog", "ml_platform", "Unity Catalog")
dbutils.widgets.text("schema", "production", "Schema")
dbutils.widgets.text("run_date", "", "Run Date (YYYY-MM-DD)")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
run_date = dbutils.widgets.get("run_date") or str(
    spark.sql("SELECT current_date()").collect()[0][0]
)

print(f"Catalog: {catalog}, Schema: {schema}, Run Date: {run_date}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set Up Unity Catalog

# COMMAND ----------

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema}")
spark.sql(f"USE SCHEMA {schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Auto Loader — Ingest Raw Files to Bronze

# COMMAND ----------

# Auto Loader configuration for cloud storage ingestion
# In production, point source_path to your actual S3/ADLS/GCS bucket
AUTO_LOADER_CONFIG = {
    "cloudFiles.format": "parquet",  # or csv, json
    "cloudFiles.schemaLocation": f"/mnt/checkpoints/{schema}/bronze_schema",
    "cloudFiles.inferColumnTypes": "true",
    "cloudFiles.schemaEvolutionMode": "addNewColumns",
}

# Source path (replace with actual cloud storage path)
source_path = f"/mnt/raw-data/{schema}/incoming/"
bronze_table = f"{catalog}.{schema}.bronze_raw_data"

# Read with Auto Loader (Structured Streaming with file notification)
# df_stream = (
#     spark.readStream
#     .format("cloudFiles")
#     .options(**AUTO_LOADER_CONFIG)
#     .load(source_path)
#     .withColumn("_ingestion_timestamp", current_timestamp())
#     .withColumn("_source_file", input_file_name())
#     .withColumn("_run_date", lit(run_date))
# )

# Write to Bronze Delta table
# (
#     df_stream.writeStream
#     .format("delta")
#     .option("checkpointLocation", f"/mnt/checkpoints/{schema}/bronze")
#     .outputMode("append")
#     .trigger(availableNow=True)  # Process all available files then stop
#     .toTable(bronze_table)
# )

# ── For Demo: Generate synthetic data directly ──
from pyspark.sql.functions import (
    col, current_timestamp, lit, rand, randn, when, expr
)
import pyspark.sql.functions as F

NUM_ROWS = 10_000_000  # 10M training records per model
NUM_FEATURES = 50

# Generate synthetic data
df = spark.range(0, NUM_ROWS).toDF("record_id")

# Add numeric features
for i in range(30):
    df = df.withColumn(f"num_{i:03d}", randn(seed=42 + i) * 10 + rand(seed=i))

# Add categorical features
categories = {
    "cat_geography": ["US", "EU", "APAC", "LATAM", "MEA"],
    "cat_segment": ["retail", "wholesale", "enterprise", "smb"],
    "cat_channel": ["online", "store", "phone", "partner", "direct"],
}
for col_name, values in categories.items():
    n_cats = len(values)
    df = df.withColumn(
        col_name,
        expr(f"element_at(array({','.join(repr(v) for v in values)}), "
             f"int(rand(seed={hash(col_name) % 1000}) * {n_cats}) + 1)")
    )

# Add target columns
df = df.withColumn(
    "target_binary",
    when(col("num_000") + col("num_001") + randn(seed=99) > 0, 1).otherwise(0)
)
df = df.withColumn(
    "target_continuous",
    col("num_000") * 5 + col("num_002") * 3 + randn(seed=100) * 10
)

# Add metadata
df = (
    df.withColumn("_ingestion_timestamp", current_timestamp())
    .withColumn("_run_date", lit(run_date))
)

# Write to Bronze Delta table
df.write.format("delta").mode("overwrite").saveAsTable(bronze_table)

print(f"✅ Bronze table created: {bronze_table}")
print(f"   Rows: {spark.table(bronze_table).count():,}")
print(f"   Columns: {len(spark.table(bronze_table).columns)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Quality Checks (DLT Expectations Pattern)

# COMMAND ----------

from pyspark.sql.functions import count, sum as spark_sum, isnull

bronze_df = spark.table(bronze_table)

# Expectation 1: No null record_ids
null_ids = bronze_df.filter(isnull("record_id")).count()
assert null_ids == 0, f"EXPECT record_id IS NOT NULL: {null_ids} nulls found"

# Expectation 2: No duplicate record_ids
total = bronze_df.count()
distinct = bronze_df.select("record_id").distinct().count()
assert total == distinct, f"EXPECT UNIQUE record_id: {total - distinct} duplicates"

# Expectation 3: Target binary is 0 or 1
invalid_targets = bronze_df.filter(
    ~col("target_binary").isin(0, 1)
).count()
assert invalid_targets == 0, f"EXPECT target_binary IN (0,1): {invalid_targets} invalid"

print("✅ All data quality expectations passed")
print(f"   Total rows: {total:,}")
print(f"   Distinct IDs: {distinct:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Optimize Bronze Table

# COMMAND ----------

spark.sql(f"OPTIMIZE {bronze_table} ZORDER BY (record_id)")
print(f"✅ Bronze table optimized with Z-ORDER on record_id")
