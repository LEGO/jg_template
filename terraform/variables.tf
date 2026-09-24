variable "databricks_host" {
  description = "Databricks workspace host URL"
  type        = string
  default     = "https://adb-123456789.azuredatabricks.net"
}

variable "databricks_token" {
  description = "Databricks authentication token"
  type        = string
  sensitive   = true
  default     = ""
}

variable "environment" {
  description = "Deployment environment (dev, staging, prod) ensuring environment symmetry"
  type        = string
  default     = "dev"
}

variable "catalog_name" {
  description = "Unity Catalog name"
  type        = string
  default     = "sandbox"
}

variable "schema_name" {
  description = "Schema name for the ML data product"
  type        = string
  default     = "lego_ml_product"
}

variable "owner" {
  description = "Owner team email or group"
  type        = string
  default     = "ai-engineering@lego.com"
}

variable "spark_version" {
  description = "Databricks runtime version"
  type        = string
  default     = "15.4.x-cpu-ml-scala2.12"
}

variable "node_type_id" {
  description = "Worker node instance type"
  type        = string
  default     = "Standard_D4ds_v5"
}

variable "num_workers" {
  description = "Number of cluster worker nodes"
  type        = number
  default     = 2
}
