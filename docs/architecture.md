# Architecture Documentation

## System Overview

This template assumes that data is already ingested into Unity Catalog. If that is not the case please have a look at [LEGO Nexus workflow generator](https://github.com/LEGO/nexus-workflow-generator).

The template sets up a mocked ML pipeline using sandbox data on Databricks that enables LEGO set piece-count prediction both on a managed serving endpoint and in batch inference runs.

**Core flow:** LEGO Sets Bronze data → Feature Engineering → Delta tables → Model Training → REST API + Batch Inference

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
- **Entry:** [`data_preprocessing.py`](../src/lego_parts_predictor/data_preprocessing.py)
- **Job:** [`data_processing.yml`](../resources/data_processing.yml)
- Reads `lego_sets_bronze` from Unity Catalog, drops invalid/theme-only rows, casts to numerics, filters to sets with `parts > 0`, dedupes, builds the theme vocabulary (top 100 + `Other`), optionally windows by release year, one-hot encodes `theme_name`, and logs the dataset to MLflow.
- **Output:** `lego_set_features` Delta table (and, on the first run, `lego_set_features_baseline`)
- **Schedule:** Weekly, Mondays 04:00 CET

### 2. Model Training
- **Entry:** [`train_model.py`](../src/lego_parts_predictor/model/train_model.py)
- **Job:** [`model_training.yml`](../resources/model_training.yml)
- Trains a HistGradientBoosting regression model on the year and one-hot theme features to predict a set's piece count. Logs parameters, metrics, and the model artifact to MLflow. Registers the model and sets the `champion` alias.
- **Cluster:** CPU (i3.2xlarge, Runtime 17.3.x ML CPU)
- **Schedule:** Weekly, Mondays 06:00 CET
- **Output:** Registered model in MLflow Registry with `champion` alias

### 3. Model Serving
- **Entry:** [`serve.py`](../src/lego_parts_predictor/model/serve.py)
- **Job:** [`model_training.yml`](../resources/model_training.yml) (`serve_model` task, depends on `train_model`)
- Deploys the champion model to a Databricks serving endpoint. Configures access permissions and workload size.
- **Compute:** CPU-optimized serving endpoint

### 4. Batch Prediction
- **Entry:** [`batch_prediction.py`](../src/lego_parts_predictor/batch_prediction.py)
- **Job:** [`model_training.yml`](../resources/model_training.yml) (`batch_inference_task`, depends on `train_model`)
- Loads the `champion` model from the MLflow Registry, runs predictions on a random subset of the feature table, and upserts results into the predictions Delta table.
- **Output:** `lego_parts_predictor_batch_predictions` Delta table

### 5. Prediction Drift Monitoring
- **Entry:** [`unpack_inference_table.py`](../src/lego_parts_predictor/monitoring/unpack_inference_table.py), [`setup_monitor.py`](../src/lego_parts_predictor/monitoring/setup_monitor.py)
- **Job:** [`model_monitoring.yml`](../resources/lego_parts_predictor/model_monitoring.yml) (`unpack_inference_table` → `setup_monitor`)
- Flattens the serving endpoint's `<endpoint>_payload` AI Gateway inference table into a typed `*_predictions_unpacked` Delta table (one successful row per scored record; error rows with `status_code != 200` are dropped), then creates/updates a managed Lakehouse Monitoring `InferenceLog` monitor on it. Each row keeps its input feature vector (`features`, zipped positionally with the prediction) alongside `Predicted_number_of_parts`. Databricks generates the profile-metrics table, drift-metrics table, and a monitoring dashboard. `model_version` is the payload table's `served_entity_id`, so drift can be sliced per served model version.
- **Schedule:** Daily, 07:00 CET (aligned with the `1 day` monitor granularity)
- **Output:** `lego_parts_predictor_predictions_unpacked` Delta table + managed monitor assets
- **Scope:** Prediction drift only; input-feature drift is handled separately.

**Known limitations — this job cannot be demonstrated in one sitting.** Unlike the data-drift
replay above, prediction drift needs real endpoint traffic accumulated over time:

- **No traffic means an empty table, not a failure.** The `<endpoint>_payload` table is created
  by AI Gateway only on its first flush after a real request, so on a fresh endpoint
  `unpack_inference_table` logs "does not exist yet (no traffic)" and writes an empty
  `*_predictions_unpacked` table so `setup_monitor` has a target. The job goes green having
  monitored nothing.
- **Payload logging is asynchronous** and lazily creates the table — allow up to an hour after
  the first request, not minutes.
- **Failed requests are invisible.** Rows with `status_code != 200` are dropped, so a request
  rejected for the wrong input shape never becomes a monitored prediction. The model is signed
  as an unnamed tensor, so the body must be `{"inputs": [[...]]}` with exactly one value per
  feature (101 in this example) in feature-table column order, id and target excluded.
- **Drift needs two windows.** `granularities=["1 day"]` puts every request made on the same
  day into one window, and drift is a comparison between windows — so traffic must span at
  least two days before the drift columns populate. There is no equivalent of the year-window
  replay for this; faking it would mean fabricating `prediction_ts` values.

For a same-session demonstration of drift, use the feature-table replay above.

---

## Data Architecture

### Unity Catalog Structure

```
my_ml_product (catalog)
├── dev_general_resources (schema)          # NOT model-specific: shared/curated tables, lookups
│   └── ...
└── dev_lego_parts_predictor_model (schema)  # everything for one model
    ├── lego_set_features (Delta)         # Features prepped for this model
    ├── lego_set_features_baseline (Delta)  # Fixed reference snapshot for drift monitoring
    ├── lego_parts_predictor (UC model)
    ├── lego_parts_predictor_batch_predictions (Delta)  # Batch inference output
    ├── lego_parts_predictor_predictions_unpacked (Delta)  # Flattened predictions for drift monitoring
    └── <monitor profile/drift metric tables>
```

Schema naming is `<env>_<...>`, where `<env>` is `dev` / `qa` / `prod`. The `local` target maps
to `dev` and appends the developer's short username (`${workspace.current_user.externalId}`), e.g.
`my_ml_product.dev_dkAndrMo_general_resources` and
`my_ml_product.dev_dkAndrMo_lego_parts_predictor_model`.

Raw source data lives outside the bundle (default `ai_agency.general_resource.lego_sets_bronze`).

### Key Schemas

**lego_sets_bronze:**
```
set_number: string
set_name: string           # not used by the pipeline
year_released: string      # text with a trailing '.0', e.g. "1965.0"
number_of_parts: string    # text with a trailing '.0', e.g. "43.0"
image_url: string          # not used by the pipeline
theme_name: string
```

**lego_set_features:**
```
set_number: string
number_of_parts: float
year_released: int
Technic: int
Star_Wars: int
...
Other: int                 # see note below
```

`Other` holds both the source's own `Other` theme and every theme outside the top 100
(23.4% of rows combined). The residual bucket is deliberately merged with the real
theme, which is itself a miscellaneous category.

**lego_parts_predictor_batch_predictions:** as `lego_set_features`, plus
`Predicted_number_of_parts: bigint`.

**Why the batch table is an integer and the serving endpoint returns a float.** This
asymmetry is deliberate, not an oversight. A LEGO set has a whole, non-negative number of
pieces, so `batch_prediction.py` rounds the regressor's raw output and clips it at zero
before writing — on the current data a few predictions come out negative, and a set with
-12 pieces in a curated table is worse than useless. The serving endpoint deliberately
does **not** round: a regressor's contract is a point estimate on a continuous scale, and
`204.37` honestly conveys "about 204" where `204` implies precision the model does not
have. Rounding inside the model would also make one consumer's formatting preference
everyone's contract, and would degrade the drift monitor specifically — the `InferenceLog`
monitor computes distribution statistics over `Predicted_number_of_parts`, and quantising
predictions before it sees them coarsens exactly the signal it exists to track (for small
sets, rounding to 0 versus 1 is a large *relative* distortion). Callers that want an
integer round at the point of use.

**lego_parts_predictor_predictions_unpacked:**
```
record_id: string
prediction_ts: timestamp
Predicted_number_of_parts: double
features: array<double>   # input feature vector (bare positional array, no theme names)
model_version: string
```

---

## Model Architecture

### HistGradientBoosting Regression

- **Algorithm:** Histogram-based gradient boosting (`HistGradientBoostingRegressor`) via scikit-learn
- **Target:** `number_of_parts` (piece count of a LEGO set, float; rounded to a non-negative integer for batch predictions)
- **Features:** `year_released` plus one-hot encoded theme columns (all columns except `set_number` and `number_of_parts`) — 101 features
- **Config:** [`model_config.yml`](../src/lego_parts_predictor/model/model_config.yml)
  ```yaml
  default_hgb:
    max_iter: 200
    learning_rate: 0.1
    max_depth: 6
    min_samples_leaf: 20
    l2_regularization: 0.0
    random_state: 42
  ```
- **Why not AdaBoost:** the example originally used `AdaBoostRegressor`, which on this data
  is the only model measured that performs *worse than predicting a constant*. AdaBoost.R2
  reweights toward the examples it fits worst — here the handful of 5,000-11,695-piece sets
  — dragging every prediction upward. Measured on the same 20% holdout (seed 42):

  | Model | RMSE | vs baseline |
  |---|---|---|
  | predict the training mean | 476 | — |
  | AdaBoost `n_estimators=50, lr=1.0` | 530 | +11% (worse) |
  | AdaBoost `n_estimators=200, lr=0.05` | 579 | +22% (tuning made it worse) |
  | RandomForest `n=300` | 452 | −5% |
  | Ridge `alpha=1.0` | 441 | −7% |
  | **HistGradientBoosting** | **431** | **−9%** |

  A log-transformed target reaches a better MAE (157 vs 184) but a worse RMSE, and would
  require inverse-transforming in training, batch prediction and serving — so the target
  is left untransformed.

- **Metrics logged:** `rmse`, plus `baseline_rmse` (predict-the-mean) and
  `skill_vs_baseline` on a 20% holdout, so a model with negative skill is visible rather
  than hidden behind a lone RMSE figure
- **Registry:** MLflow Model Registry (Unity Catalog) with `champion` alias

---

## Deployment

### Databricks Bundle

**Config:** [`databricks.yml`](../databricks.yml)

**Targets:** dev, qa, prod (environment-specific configs)

**Key variables (dev):**
```yaml
catalog: <data_product_name>   # defaults to the data product name; must be an existing UC catalog
schema_prefix: dev             # local: dev_<short_name>
general_schema: ${var.schema_prefix}_general_resources
<data_product_name>_model_schema: ${var.schema_prefix}_${var.<data_product_name>_model_name}_model
model_name: lego_parts_predictor
feature_store_table_name: lego_set_features
batch_prediction_table: lego_parts_predictor_batch_predictions
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

## Demonstrating data drift

The dataset is a static 2023 snapshot, but it spans 1949–2023 and LEGO sets changed a great
deal over that period. Replaying those eras through the preprocessing job's year window
produces real, measurable drift with no synthetic noise.

### 1. Reset both tables first

The feature table is written with a `MERGE`, which inserts and updates but **never deletes** —
so a narrower year window cannot shrink a table that already holds more rows. If the pipeline
has ever run with the default empty window, drop both tables or every step below will change
nothing:

```sql
DROP TABLE <catalog>.<schema>.lego_set_features;
DROP TABLE <catalog>.<schema>.lego_set_features_baseline;
```

Dropping the baseline alone is not enough: it is written from the in-memory windowed frame, so
it stays correct while the feature table goes stale. Then delete the monitor so the pipeline
recreates it against the new table:

```bash
databricks quality-monitors delete <catalog>.<schema>.lego_set_features --profile <profile>
```

### 2. Move the window, one era at a time

`min_year`/`max_year` are `python_wheel_task.named_parameters`, baked in at deploy time, so
`bundle run --params` cannot reach them — the window moves by redeploying with `--var`. The
variables are named `<data_product_name>_min_year` / `_max_year`; confirm yours with
`grep _min_year databricks.yml`, since a wrong name fails with `variable ... has not been
defined`.

```bash
# 1. Baseline era. This run also creates the baseline table.
databricks bundle deploy -t local --var="catalog=<your_catalog>" \
  --var="<data_product_name>_min_year=1949" --var="<data_product_name>_max_year=1999"
databricks bundle run data_preprocessing_job -t local

# 2. Advance the window, redeploy, re-run.
databricks bundle deploy -t local --var="catalog=<your_catalog>" \
  --var="<data_product_name>_min_year=2000" --var="<data_product_name>_max_year=2009"
databricks bundle run data_preprocessing_job -t local

# 3. Advance again.
databricks bundle deploy -t local --var="catalog=<your_catalog>" \
  --var="<data_product_name>_min_year=2010" --var="<data_product_name>_max_year=2023"
databricks bundle run data_preprocessing_job -t local
```

`catalog` defaults to your data product name and the bundle creates schemas but **not** the
catalog, so `<your_catalog>` must already exist. Each monitor refresh takes about 7 minutes,
so wait before checking results.

### 3. Where to look for the drift

**In the monitoring dashboard:** the `Numerical Distribution Change` section, using
*Filter by Column Name* → `number_of_parts` (the profile tables run to 13 pages of theme
columns otherwise). `Row Count Over Time` should step 4,473 → 8,191 → 17,112, and after a
clean reset the default consecutive-refresh comparison shows drift without editing any widget.

**In SQL**, which is unambiguous and needs no navigation:

```sql
-- Distribution per refresh. log_type='BASELINE' is the reference; 'INPUT' rows are refreshes.
SELECT window.start, log_type, count, round(avg,2) AS avg_parts
FROM <catalog>.<schema>.lego_set_features_profile_metrics
WHERE column_name = 'number_of_parts' AND slice_key IS NULL ORDER BY window.start;

-- Drift. BASELINE compares to the fixed reference; CONSECUTIVE to the previous refresh.
SELECT window.start, drift_type, count_delta, round(avg_delta,2) AS avg_delta,
       round(wasserstein_distance,2) AS wass, ks_test.pvalue AS ks_p
FROM <catalog>.<schema>.lego_set_features_drift_metrics
WHERE column_name = 'number_of_parts' AND slice_key IS NULL ORDER BY window.start;
```

Measured on a live workspace after the three steps above:

| Step | Rows | Mean parts | `avg_delta` vs baseline | Wasserstein | KS p-value |
|---|---|---|---|---|---|
| 1 · 1949–1999 | 4,473 | 105.90 | 0.00 | 0.00 | 1.0 |
| 2 · +2000–2009 | 8,191 | 137.27 | +31.37 | 31.56 | 2.9e-4 |
| 3 · +2010–2023 | 17,112 | 204.08 | **+98.19** | 99.24 | **0.0** |

Step 1 correctly shows zero — baseline and current are the same snapshot. The theme mix
shifts sharply too (Classic Town 8.8% → 0.0%, Star Wars 0.3% → 5.6%); use `js_distance` for
those, since it is `NULL` for numeric columns and `wasserstein_distance` is `NULL` for
categorical ones.

### Two things to ignore

`set_number` is always flagged in the categorical section: Lakehouse Monitoring profiles every
column with no exclusion option, and a primary key's values are unique by definition, so its
chi-squared test always reports significance (p ~ 1e-12). On a fresh monitor it can be the
only flagged column, which reads like a finding and is not.

**Known limitations of the example:** the dataset is a static 2023 snapshot (Rebrickable is
the upstream refresh path if a live source is ever wanted); the target is heavily right-skewed
(median 54, mean 204, max 11,695), so RMSE is dominated by a few very large sets; `Other` is a
large 23.4% bucket; the model is undefined for the zero-part merchandise that is filtered out;
and drift only appears when the year window is moved deliberately.

---

## Technology Stack

| Component           | Technology                 |
| ------------------- | -------------------------- |
| **Platform**        | Databricks                 |
| **Compute**         | Apache Spark               |
| **Storage**         | Unity Catalog (Delta Lake) |
| **ML Framework**    | scikit-learn (HistGradientBoosting) |
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
- [scikit-learn HistGradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html)
- [OmegaConf](https://omegaconf.readthedocs.io/)
