# Developer Guide

Practical examples and recipes for working with the MLOps pipeline.

## Table of Contents
1. [Local Development Setup](#local-development-setup)
2. [Running Pipelines](#running-pipelines)
3. [MLflow & Model Registry Recipes](#mlflow--model-registry-recipes)
4. [Query Unity Catalog Tables](#query-unity-catalog-tables)
5. [Test Serving Endpoint via REST API](#test-serving-endpoint-via-rest-api)
6. [Debugging Tips](#debugging-tips)

---

## Local Development Setup

### Install dependencies

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Navigate to project root
cd /path/to/mlops_template

# Install project dependencies (including dev extras)
uv sync --all-extras

# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows
```

### Configure Databricks CLI

```bash
# Configure with OAuth (recommended)
databricks configure

# Provide:
# - Databricks workspace URL: https://lego-ssc-dev.cloud.databricks.com
# - OAuth or personal access token

# Test connection
databricks workspace list
```

---

## Running Pipelines

### Deploy and run via CLI

```bash
# Deploy bundle to dev
databricks bundle deploy --target dev --profile lego-ssc-dev

# Run a specific job interactively (choose from list)
databricks bundle run --target dev --profile lego-ssc-dev

# Or run a specific job by name
databricks bundle run data_preprocessing_job --target dev --profile lego-ssc-dev
databricks bundle run model_training_job --target dev --profile lego-ssc-dev
```

### Use an existing cluster (faster iteration)

By default, jobs create new clusters which take 5–10 minutes to start. To speed up development, override with an existing cluster:

1. Open Databricks workspace → **Compute**
2. Copy the **Cluster ID** from the URL or details page

```bash
databricks bundle run model_training_job --target dev \
  --var="existing_cluster_id=0123-456789-abcdef01"
```

> **Note:** The bundle automatically builds and packages the wheel via `uv build` during `bundle deploy`. Manual `uv build` is not needed.

### Run tests locally

```bash
pytest tests/
```

---

## MLflow & Model Registry Recipes

### Load the champion model locally

```python
import mlflow

model = mlflow.pyfunc.load_model(
    "models:/my_ml_product.dev_lego_parts_predictor_model.lego_parts_predictor@champion"
)

# Run a prediction (expects a numpy array of year + one-hot theme features)
import numpy as np
features = np.array([[2005, 1, 0, 1, 0, 0, 1, 0, 0]])  # year_released, one-hot theme columns
predictions = model.predict(features)
print(predictions)  # e.g. [312.0]
```

### Update the champion alias manually

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Promote version 5 to champion
client.set_registered_model_alias(
    name="my_ml_product.dev_lego_parts_predictor_model.lego_parts_predictor",
    alias="champion",
    version=5,
)
```

### Compare model versions by RMSE

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# List all versions of the model
versions = client.search_model_versions(
    "name='my_ml_product.dev_lego_parts_predictor_model.lego_parts_predictor'"
)

for v in versions:
    run = client.get_run(v.run_id)
    rmse = run.data.metrics.get("rmse", "N/A")
    print(f"Version {v.version} | Alias: {v.aliases} | RMSE: {rmse}")
```

### Start an MLflow experiment run manually

```python
from common.mlflow_helper import start_mlflow_experiment_and_run

run = start_mlflow_experiment_and_run(
    experiment_path="/ai_agency/mlops_pipeline/dev/<your_user>/lego_parts_predictor_model_training"
)
```

---

## Query Unity Catalog Tables

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

# Inspect raw data
bronze = spark.read.table("ai_agency.general_resource.lego_sets_bronze")
bronze.printSchema()
bronze.show(5)

# Inspect feature table
features = spark.read.table("my_ml_product.dev_lego_parts_predictor_model.lego_set_features")
features.printSchema()

# Check theme column distribution
import pyspark.sql.functions as F
features.select([F.sum(c).alias(c) for c in features.columns if c not in ("set_number", "number_of_parts")]).show()

# Inspect batch predictions
preds = spark.read.table("my_ml_product.dev_lego_parts_predictor_model.lego_parts_predictor_batch_predictions")
preds.select("set_number", "number_of_parts", "Predicted_number_of_parts").show(10)
```

---

## Test Serving Endpoint via REST API

```python
import requests

def query_lego_parts_endpoint(features: list[float], token: str) -> float:
    """Query the LEGO set-size predictor serving endpoint.

    The model is signed as an unnamed tensor (see train_model.py), so the endpoint
    takes the `inputs` tensor format — a positional feature vector, not a named
    column format.

    Args:
        features: Positional feature vector, e.g. [2005, 1, 0, 1, 0, 0, 1, 0, 0].
                  Values must be in feature-table column order (year_released, then
                  the one-hot theme columns; id and target columns excluded) and
                  must have the trained feature count — see serve.py's deployment
                  log for the exact column order for a given model.
        token: Databricks personal access token

    Returns:
        Predicted piece count as a float.
    """
    workspace_url = "https://lego-ssc-dev.cloud.databricks.com"
    endpoint_name = "lego_parts_predictor_endpoint"

    response = requests.post(
        f"{workspace_url}/serving-endpoints/{endpoint_name}/invocations",
        headers={"Authorization": f"Bearer {token}"},
        json={"inputs": [features]},
    )

    response.raise_for_status()
    return response.json()["predictions"][0]


# Example usage: year_released, then one-hot theme columns, positionally
parts = query_lego_parts_endpoint(
    features=[2005, 1, 0, 1, 0, 0, 1, 0, 0],
    token="dapi...",
)
print(f"Predicted parts: {parts:.0f}")
```

---

## Debugging Tips

### Schema mismatch on upsert

If `upsert_delta_table` fails with `DELTA_MERGE_UNRESOLVED_EXPRESSION`, the target table schema is out of sync with the source. The helper automatically handles this by overwriting with the new schema when the primary key column is missing. If you want to force a clean overwrite manually:

```python
dataframe.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("my_ml_product.dev_lego_parts_predictor_model.lego_set_features")
```

### Inspect MLflow run parameters and metrics

```python
import mlflow

client = mlflow.MlflowClient()

# Get the run linked to the champion model version
model_version = client.get_model_version_by_alias(
    name="my_ml_product.dev_lego_parts_predictor_model.lego_parts_predictor",
    alias="champion",
)
run = client.get_run(model_version.run_id)

print("Params:", run.data.params)
print("Metrics:", run.data.metrics)
```

### Re-run preprocessing only (no model training)

```bash
databricks bundle run data_preprocessing_job --target dev --profile lego-ssc-dev
```

### Check active MLflow runs

```python
import mlflow

runs = mlflow.search_runs(
    experiment_names=["/ai_agency/mlops_pipeline/dev/<your_user>/lego_parts_predictor_model_training"],
    order_by=["start_time DESC"],
    max_results=5,
)
print(runs[["run_id", "status", "metrics.rmse", "start_time"]])
```

---

For architecture details, see [architecture.md](architecture.md).  
For data flow diagrams, see [data-flow.md](data-flow.md).
