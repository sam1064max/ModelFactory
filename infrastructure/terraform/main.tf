# ──────────────────────────────────────────────────────────────────────────────
# Terraform — Databricks MLOps Platform Infrastructure
# ──────────────────────────────────────────────────────────────────────────────
# This Terraform configuration provisions the Databricks workspace resources
# needed for the MLOps platform:
#   - Unity Catalog (catalog, schemas, grants)
#   - Cluster Policies (cost controls)
#   - Instance Pools (pre-warmed compute)
#   - Jobs/Workflows (orchestration)
#   - Secrets (credentials)
#
# Usage:
#   terraform init
#   terraform plan -var-file="prod.tfvars"
#   terraform apply -var-file="prod.tfvars"
# ──────────────────────────────────────────────────────────────────────────────

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    databricks = {
      source  = "databricks/databricks"
      version = "~> 1.40.0"
    }
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  backend "s3" {
    bucket         = "ml-platform-terraform-state"
    key            = "databricks/mlops-platform/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "terraform-state-lock"
    encrypt        = true
  }
}

# ── Providers ────────────────────────────────────────────────────────────────

provider "databricks" {
  host  = var.databricks_workspace_url
  token = var.databricks_token
}

provider "aws" {
  region = var.aws_region
}

# ── Unity Catalog ────────────────────────────────────────────────────────────

resource "databricks_catalog" "ml_platform" {
  name    = var.catalog_name
  comment = "MLOps Platform catalog for model training, inference, and monitoring"

  properties = {
    purpose    = "ml_platform"
    managed_by = "terraform"
  }
}

resource "databricks_schema" "production" {
  catalog_name = databricks_catalog.ml_platform.name
  name         = "production"
  comment      = "Production schema for ML models and features"
}

resource "databricks_schema" "staging" {
  catalog_name = databricks_catalog.ml_platform.name
  name         = "staging"
  comment      = "Staging schema for model validation"
}

resource "databricks_schema" "monitoring" {
  catalog_name = databricks_catalog.ml_platform.name
  name         = "monitoring"
  comment      = "Schema for drift metrics, alerts, and monitoring data"
}

# ── Unity Catalog Grants ─────────────────────────────────────────────────────

resource "databricks_grants" "catalog_grants" {
  catalog = databricks_catalog.ml_platform.name

  grant {
    principal  = var.ml_engineer_group
    privileges = ["USE_CATALOG", "USE_SCHEMA", "SELECT", "MODIFY", "CREATE_TABLE", "CREATE_FUNCTION"]
  }

  grant {
    principal  = var.data_scientist_group
    privileges = ["USE_CATALOG", "USE_SCHEMA", "SELECT", "CREATE_TABLE"]
  }

  grant {
    principal  = var.auditor_group
    privileges = ["USE_CATALOG", "USE_SCHEMA", "SELECT"]
  }

  grant {
    principal  = var.service_principal_name
    privileges = ["USE_CATALOG", "USE_SCHEMA", "SELECT", "MODIFY", "CREATE_TABLE", "CREATE_FUNCTION"]
  }
}

# ── Cluster Policies ─────────────────────────────────────────────────────────

resource "databricks_cluster_policy" "training_policy" {
  name = "ml-platform-training-policy"

  definition = jsonencode({
    "spark_version" : {
      "type" : "fixed",
      "value" : var.spark_version
    },
    "node_type_id" : {
      "type" : "allowlist",
      "values" : ["i3.2xlarge", "i3.4xlarge", "i3.8xlarge", "r5.2xlarge"]
    },
    "autoscale.min_workers" : {
      "type" : "range",
      "minValue" : 2,
      "maxValue" : 10
    },
    "autoscale.max_workers" : {
      "type" : "range",
      "minValue" : 10,
      "maxValue" : var.max_training_workers
    },
    "aws_attributes.availability" : {
      "type" : "fixed",
      "value" : "SPOT_WITH_FALLBACK"
    },
    "custom_tags.team" : {
      "type" : "fixed",
      "value" : "ml-platform"
    },
    "custom_tags.cost_center" : {
      "type" : "fixed",
      "value" : var.cost_center
    },
    "autotermination_minutes" : {
      "type" : "range",
      "minValue" : 10,
      "maxValue" : 60
    }
  })
}

resource "databricks_cluster_policy" "inference_policy" {
  name = "ml-platform-inference-policy"

  definition = jsonencode({
    "spark_version" : {
      "type" : "fixed",
      "value" : var.spark_version
    },
    "node_type_id" : {
      "type" : "allowlist",
      "values" : ["c5.4xlarge", "c5.9xlarge", "c5.18xlarge"]
    },
    "autoscale.min_workers" : {
      "type" : "range",
      "minValue" : 10,
      "maxValue" : 30
    },
    "autoscale.max_workers" : {
      "type" : "range",
      "minValue" : 30,
      "maxValue" : var.max_inference_workers
    },
    "aws_attributes.availability" : {
      "type" : "fixed",
      "value" : "SPOT_WITH_FALLBACK"
    },
    "spark_conf.spark.databricks.photon.enabled" : {
      "type" : "fixed",
      "value" : "true"
    }
  })
}

# ── Instance Pools ───────────────────────────────────────────────────────────

resource "databricks_instance_pool" "training_pool" {
  instance_pool_name = "ml-platform-training-pool"
  min_idle_instances = var.training_pool_min_idle
  max_capacity       = var.training_pool_max_capacity
  node_type_id       = "i3.2xlarge"

  idle_instance_autotermination_minutes = 30

  aws_attributes {
    availability = "SPOT_WITH_FALLBACK"
    spot_bid_price_percent = 100
  }

  custom_tags = {
    team        = "ml-platform"
    purpose     = "model-training"
    cost_center = var.cost_center
  }
}

resource "databricks_instance_pool" "inference_pool" {
  instance_pool_name = "ml-platform-inference-pool"
  min_idle_instances = var.inference_pool_min_idle
  max_capacity       = var.inference_pool_max_capacity
  node_type_id       = "c5.4xlarge"

  idle_instance_autotermination_minutes = 20

  aws_attributes {
    availability = "SPOT_WITH_FALLBACK"
    spot_bid_price_percent = 100
  }

  custom_tags = {
    team        = "ml-platform"
    purpose     = "batch-inference"
    cost_center = var.cost_center
  }
}

# ── Secrets Scope ────────────────────────────────────────────────────────────

resource "databricks_secret_scope" "ml_platform" {
  name = "ml-platform"
}

resource "databricks_secret" "slack_webhook" {
  scope        = databricks_secret_scope.ml_platform.name
  key          = "slack-webhook-url"
  string_value = var.slack_webhook_url
}

resource "databricks_secret" "pagerduty_key" {
  scope        = databricks_secret_scope.ml_platform.name
  key          = "pagerduty-integration-key"
  string_value = var.pagerduty_integration_key
}

# ── S3 Buckets (Data Lake) ───────────────────────────────────────────────────

resource "aws_s3_bucket" "ml_data_lake" {
  bucket = "${var.project_name}-data-lake-${var.environment}"

  tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

resource "aws_s3_bucket_versioning" "ml_data_lake" {
  bucket = aws_s3_bucket.ml_data_lake.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "ml_data_lake" {
  bucket = aws_s3_bucket.ml_data_lake.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = var.kms_key_id
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "ml_data_lake" {
  bucket = aws_s3_bucket.ml_data_lake.id

  rule {
    id     = "archive-bronze-data"
    status = "Enabled"

    filter {
      prefix = "bronze/"
    }

    transition {
      days          = 90
      storage_class = "GLACIER"
    }
  }

  rule {
    id     = "expire-monitoring-reports"
    status = "Enabled"

    filter {
      prefix = "monitoring/"
    }

    expiration {
      days = 365
    }
  }
}
