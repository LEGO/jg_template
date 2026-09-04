# Architecture Documentation

## System Overview

This template assumes that data is already ingested into Unity Catalog. If that is not the case please have a look at [LEGO Nexus workflow generator](https://github.com/LEGO/nexus-workflow-generator).

The template sets up a mocked ML pipeline using sandbox data on Databricks that enables anime score prediction both on a managed serving endpoint and in batch inference runs.

**Core flow:** Anime Bronze data → Feature Engineering → Delta tables → Model Training → REST API + Batch Inference

### Architecture Principles

1. **Separation of concerns** – Each pipeline stage is independent
2. **Data versioning** – Unity Catalog with Delta Lake
3. **Model versioning** – MLflow Model Registry
4. **Configuration as code** – Databricks bundles (YAML)
5. **Scalability** – Spark for data processing, scikit-learn for training

---

## Pipeline Components

Four sequential stages:

### 1. Data Preprocessing
- **Entry:** [`data_preprocessing.py`](../src/anime_score_predictor/data_preprocessing.py)
- **Job:** [`data_processing.yml`](../resources/data_processing.yml)
- Reads `anime_bronze` from Unity Catalog, casts `Score` to float, one-hot encodes genres, and logs the dataset to MLflow.
- **Output:** `anime_features` Delta table
- **Schedule:** Weekly, Mondays 04:00 CET

### 2. Model Training
- **Entry:** [`train_model.py`](../src/anime_score_predictor/model/train_model.py)
- **Job:** [`model_training.yml`](../resources/model_training.yml)
- Trains a Lasso regression model on genre features to predict anime scores. Logs parameters, metrics, and the model artifact to MLflow. Registers the model and sets the `champion` alias.
- **Cluster:** CPU (i3.2xlarge, Runtime 17.3.x ML CPU)
- **Schedule:** Weekly, Mondays 06:00 CET
- **Output:** Registered model in MLflow Registry with `champion` alias

### 3. Model Serving
- **Entry:** [`serve.py`](../src/anime_score_predictor/model/serve.py)
- **Job:** [`model_training.yml`](../resources/model_training.yml) (`serve_model` task, depends on `train_model`)
- Deploys the champion model to a Databricks serving endpoint. Configures access permissions and workload size.
- **Compute:** CPU-optimized serving endpoint

### 4. Batch Prediction
- **Entry:** [`batch_prediction.py`](../src/anime_score_predictor/batch_prediction.py)
- **Job:** [`model_training.yml`](../resources/model_training.yml) (`batch_inference_task`, depends on `train_model`)
- Loads the `champion` model from the MLflow Registry, runs predictions on a random subset of the feature table, and upserts results into the predictions Delta table.
- **Output:** `anime_score_predictor_batch_predictions` Delta table

### 5. Prediction Drift Monitoring
- **Entry:** [`unpack_inference_table.py`](../src/anime_score_predictor/monitoring/unpack_inference_table.py), [`setup_monitor.py`](../src/anime_score_predictor/monitoring/setup_monitor.py)
- **Job:** [`model_monitoring.yml`](../resources/anime_score_predictor/model_monitoring.yml) (`unpack_inference_table` → `setup_monitor`)
- Flattens the serving endpoint's `<endpoint>_payload` AI Gateway inference table into a typed `*_predictions_unpacked` Delta table (one successful row per scored record; error rows with `status_code != 200` are dropped), then creates/updates a managed Lakehouse Monitoring `InferenceLog` monitor on it. Each row keeps its input feature vector (`features`, zipped positionally with the prediction) alongside `Predicted_Score`. Databricks generates the profile-metrics table, drift-metrics table, and a monitoring dashboard. `model_version` is the payload table's `served_entity_id`, so drift can be sliced per served model version.
- **Schedule:** Daily, 07:00 CET (aligned with the `1 day` monitor granularity)
- **Output:** `anime_score_predictor_predictions_unpacked` Delta table + managed monitor assets
- **Scope:** Prediction drift only; input-feature drift is handled separately.

---

## Data Architecture

### Unity Catalog Structure

```
my_ml_product (catalog)
├── dev_general_resources (schema)          # NOT model-specific: shared/curated tables, lookups
│   └── ...
└── dev_anime_score_predictor_model (schema)  # everything for one model
    ├── anime_features (Delta)         # Features prepped for this model
    ├── anime_score_predictor (UC model)
    ├── anime_score_predictor_batch_predictions (Delta)  # Batch inference output
    ├── anime_score_predictor_predictions_unpacked (Delta)  # Flattened predictions for drift monitoring
    └── <monitor profile/drift metric tables>
```

Schema naming is `<env>_<...>`, where `<env>` is `dev` / `qa` / `prod`. The `local` target maps
to `dev` and appends the developer's short username (`${workspace.current_user.externalId}`), e.g.
`my_ml_product.dev_dkAndrMo_general_resources` and
`my_ml_product.dev_dkAndrMo_anime_score_predictor_model`.

Raw source data lives outside the bundle (default `ai_enablement.general_resources.anime_bronze`).

### Key Schemas

**anime_bronze:**
```
MAL_ID: string
Name: string
Score: string
Genres: string
Synopsis: string
```

**anime_features:**
```
Name: string
Score: float
Seinen: int
Romance: int
...
Parody: int
Sci-Fi: int
```

**anime_score_predictor_batch_predictions:**
```
Name: string
Score: float
Seinen: int
Romance: int
...
Parody: int
Sci-Fi: int
Predicted_Score: float
```

**anime_score_predictor_predictions_unpacked:**
```
record_id: string
prediction_ts: timestamp
Predicted_Score: double
features: array<double>   # input feature vector (bare positional array, no genre names)
model_version: string
```

---

## Model Architecture

### Lasso Regression

- **Algorithm:** Lasso (L1-regularized linear regression) via scikit-learn
- **Target:** `Score` (anime rating, float)
- **Features:** One-hot encoded genre columns (all columns except `Name` and `Score`)
- **Config:** [`model_config.yml`](../src/anime_score_predictor/model/model_config.yml)
  ```yaml
  lasso:
    alpha: 1.0
    max_iter: 1000
    random_state: 42
  columns:
    target_name: Score
    id: Name
  ```
- **Metrics logged:** RMSE on 20% holdout test set
- **Registry:** MLflow Model Registry (Unity Catalog) with `champion` alias

---

## Deployment

### Databricks Bundle

**Config:** [`databricks.yml`](../databricks.yml)

**Targets:** dev, qa, prod (environment-specific configs)

**Key variables (dev):**
```yaml
catalog: my_ml_product
schema_prefix: dev            # local: dev_<short_name>
general_schema: ${var.schema_prefix}_general_resources
my_ml_product_model_schema: ${var.schema_prefix}_${var.my_ml_product_model_name}_model
model_name: anime_score_predictor
feature_store_table_name: anime_features
batch_prediction_table: anime_score_predictor_batch_predictions
experiment_path: /ai_agency/mlops_pipeline/dev/<user>/<model_name>
```

### Cluster Configurations

1. **Data preprocessing:**
   - Runtime: 17.3.x-cpu-ml-scala2.13
   - Node: i3.2xlarge
   - Autoscaling: 4–8 workers

2. **Model training, serving & batch inference:**
   - Runtime: 17.3.x-cpu-ml-scala2.13
   - Node: i3.2xlarge
   - Autoscaling: 1 worker (fixed)

### Job Dependencies

```
data_preprocessing_job
    └── task: data_processing

model_training_job
    ├── task: train_model
    ├── task: serve_model         (depends_on: train_model)
    └── task: batch_inference_task (depends_on: train_model)
```

### MLflow Integration

**Model Registry workflow:**
```
Training → Log model → Register → Set alias ("champion")
                                          ↓
                              Serving endpoint + Batch inference
```

**Model versioning:**
- Each training run creates a new registered model version
- `champion` alias is updated to point to the latest trained version
- Aliases: `champion`

---

## Technology Stack

| Component           | Technology                 |
| ------------------- | -------------------------- |
| **Platform**        | Databricks                 |
| **Compute**         | Apache Spark               |
| **Storage**         | Unity Catalog (Delta Lake) |
| **ML Framework**    | scikit-learn (Lasso)       |
| **Model Serving**   | Databricks Model Serving   |
| **ML Tracking**     | MLflow                     |
| **Config**          | OmegaConf (YAML)           |
| **IaC**             | Databricks Bundles (YAML)  |
| **Package Manager** | uv                         |
| **CI/CD**           | GitHub Actions             |

### Python Dependencies

```toml
omegaconf = ">=2.3.0"
scikit-learn = ">=1.7.2"
mlflow = ">=3.8.1"
databricks-connect = ">=15.4,<15.5"
```

---

## References

- [Databricks Bundles](https://docs.databricks.com/en/dev-tools/bundles/index.html)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [scikit-learn Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)
- [OmegaConf](https://omegaconf.readthedocs.io/)
