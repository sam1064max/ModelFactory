# ──────────────────────────────────────────────────────────────────────────────
# Terraform Outputs
# ──────────────────────────────────────────────────────────────────────────────

output "catalog_name" {
  description = "Unity Catalog name"
  value       = databricks_catalog.ml_platform.name
}

output "training_pool_id" {
  description = "Instance pool ID for training clusters"
  value       = databricks_instance_pool.training_pool.id
}

output "inference_pool_id" {
  description = "Instance pool ID for inference clusters"
  value       = databricks_instance_pool.inference_pool.id
}

output "training_policy_id" {
  description = "Cluster policy ID for training"
  value       = databricks_cluster_policy.training_policy.id
}

output "inference_policy_id" {
  description = "Cluster policy ID for inference"
  value       = databricks_cluster_policy.inference_policy.id
}

output "data_lake_bucket" {
  description = "S3 bucket name for the data lake"
  value       = aws_s3_bucket.ml_data_lake.bucket
}

output "data_lake_arn" {
  description = "S3 bucket ARN for the data lake"
  value       = aws_s3_bucket.ml_data_lake.arn
}

output "secrets_scope" {
  description = "Databricks secrets scope name"
  value       = databricks_secret_scope.ml_platform.name
}
