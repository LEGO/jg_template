terraform {
  required_version = ">= 1.5.0"
  required_providers {
    databricks = {
      source  = "databricks/databricks"
      version = "~> 1.50.0"
    }
  }
}

provider "databricks" {
  host  = var.databricks_host
  token = var.databricks_token
}

# Unity Catalog Schema for ML data product
resource "databricks_schema" "ml_schema" {
  catalog_name = var.catalog_name
  name         = var.schema_name
  comment      = "MLOps schema managed via Terraform for ${var.environment} environment"
  properties = {
    "environment" = var.environment
    "owner"       = var.owner
  }
}

# Feature store Delta volume / table storage location
resource "databricks_volume" "feature_store_volume" {
  name             = "feature_store"
  catalog_name     = var.catalog_name
  schema_name      = databricks_schema.ml_schema.name
  volume_type      = "MANAGED"
  comment          = "Managed storage for Feature Store and baseline datasets"
}

# Databricks Job for Continuous Training and Model Retraining Pipeline
resource "databricks_job" "model_training_pipeline" {
  name = "mlops_${var.environment}_model_training"

  schedule {
    quartz_cron_expression = "0 0 2 ? * MON"
    timezone_id            = "UTC"
  }

  job_cluster {
    job_cluster_key = "ml_cluster"
    new_cluster {
      spark_version = var.spark_version
      node_type_id  = var.node_type_id
      num_workers   = var.num_workers
    }
  }

  task {
    task_key = "data_preparation"
    job_cluster_key = "ml_cluster"
    spark_python_task {
      python_file = "src/data_preprocessing.py"
      parameters = [
        "--catalog_name", var.catalog_name,
        "--schema_name", databricks_schema.ml_schema.name,
      ]
    }
  }

  task {
    task_key = "train_model"
    depends_on {
      task_key = "data_preparation"
    }
    job_cluster_key = "ml_cluster"
    spark_python_task {
      python_file = "src/model/train_model.py"
      parameters = [
        "--catalog_name", var.catalog_name,
        "--schema_name", databricks_schema.ml_schema.name,
      ]
    }
  }
}
