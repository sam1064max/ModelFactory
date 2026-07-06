# ──────────────────────────────────────────────────────────────────────────────
# Terraform Variables — Example Values
# ──────────────────────────────────────────────────────────────────────────────
# Copy this file to prod.tfvars and fill in your values:
#   cp example.tfvars prod.tfvars
# ──────────────────────────────────────────────────────────────────────────────

databricks_workspace_url = "https://your-workspace.cloud.databricks.com"
databricks_account_id    = "your-account-id"
aws_region               = "us-east-1"
environment              = "prod"

# Unity Catalog
catalog_name = "ml_platform"
schema_name  = "production"

# Cluster settings
min_workers   = 2
max_workers   = 50
node_type_id  = "i3.2xlarge"
spark_version = "14.3.x-scala2.12"

# Instance pool
pool_min_idle     = 5
pool_max_capacity = 100

# S3
s3_bucket_name = "your-company-mlops-platform"

# Tags
project_tag = "mlops-platform"
owner_tag   = "ml-engineering"
