# ──────────────────────────────────────────────────────────────────────────────
# Terraform Variables
# ──────────────────────────────────────────────────────────────────────────────

# ── Databricks ───────────────────────────────────────────────────────────────

variable "databricks_workspace_url" {
  description = "Databricks workspace URL"
  type        = string
}

variable "databricks_token" {
  description = "Databricks personal access token or service principal token"
  type        = string
  sensitive   = true
}

variable "spark_version" {
  description = "Databricks Runtime version"
  type        = string
  default     = "15.4.x-scala2.12"
}

# ── Project ──────────────────────────────────────────────────────────────────

variable "project_name" {
  description = "Project name for resource naming"
  type        = string
  default     = "mlops-platform"
}

variable "environment" {
  description = "Deployment environment (dev, staging, prod)"
  type        = string
  default     = "prod"
}

variable "cost_center" {
  description = "Cost center for billing"
  type        = string
  default     = "ml-ops"
}

# ── AWS ──────────────────────────────────────────────────────────────────────

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "kms_key_id" {
  description = "KMS key ID for S3 encryption"
  type        = string
  default     = ""
}

# ── Unity Catalog ────────────────────────────────────────────────────────────

variable "catalog_name" {
  description = "Unity Catalog name"
  type        = string
  default     = "ml_platform"
}

# ── Access Control Groups ────────────────────────────────────────────────────

variable "ml_engineer_group" {
  description = "Databricks group for ML engineers"
  type        = string
  default     = "ml-engineers"
}

variable "data_scientist_group" {
  description = "Databricks group for data scientists"
  type        = string
  default     = "data-scientists"
}

variable "auditor_group" {
  description = "Databricks group for auditors (read-only)"
  type        = string
  default     = "auditors"
}

variable "service_principal_name" {
  description = "Service principal for automated pipelines"
  type        = string
  default     = "ml-platform-service-principal"
}

# ── Compute Scaling ──────────────────────────────────────────────────────────

variable "max_training_workers" {
  description = "Maximum worker nodes for training clusters"
  type        = number
  default     = 50
}

variable "max_inference_workers" {
  description = "Maximum worker nodes for inference clusters"
  type        = number
  default     = 100
}

variable "training_pool_min_idle" {
  description = "Minimum idle instances in training pool"
  type        = number
  default     = 5
}

variable "training_pool_max_capacity" {
  description = "Maximum capacity of training instance pool"
  type        = number
  default     = 100
}

variable "inference_pool_min_idle" {
  description = "Minimum idle instances in inference pool"
  type        = number
  default     = 10
}

variable "inference_pool_max_capacity" {
  description = "Maximum capacity of inference instance pool"
  type        = number
  default     = 200
}

# ── Alerting ─────────────────────────────────────────────────────────────────

variable "slack_webhook_url" {
  description = "Slack webhook URL for alerts"
  type        = string
  sensitive   = true
  default     = ""
}

variable "pagerduty_integration_key" {
  description = "PagerDuty integration key for P1/P2 alerts"
  type        = string
  sensitive   = true
  default     = ""
}
