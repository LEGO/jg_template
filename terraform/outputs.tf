output "schema_id" {
  description = "The ID of the provisioned Unity Catalog schema"
  value       = databricks_schema.ml_schema.id
}

output "training_job_id" {
  description = "The ID of the model training job"
  value       = databricks_job.model_training_pipeline.id
}
