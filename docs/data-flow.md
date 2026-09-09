# Data Flow Documentation

This document provides a high-level visualization of data flow through the pipeline.

## End-to-End Data Flow

```
┌─────────────────────────────────────────────────────┐
│  Unity Catalog: lego_sets_bronze                    │
│  • set_number: string                               │
│  • set_name: string                                 │
│  • year_released: string  (text, trailing '.0')     │
│  • number_of_parts: string  (text, trailing '.0')   │
│  • image_url: string                                │
│  • theme_name: string                               │
└────────────────────────┬────────────────────────────┘
                         │ spark.table()
                         ▼
┌─────────────────────────────────────────────────────┐
│  STAGE 1: Data Preprocessing                        │
│  • Drop theme-only / blank id-or-target rows         │
│  • Cast year_released, number_of_parts → numeric    │
│  • Filter: number_of_parts > 0 (drop merchandise)   │
│  • Dedupe on set_number                             │
│  • Build theme vocabulary (top 100 + Other)         │
│  • Apply year window (drift-replay lever)           │
│  • One-hot encode theme_name against the vocabulary │
│  • Select: set_number, number_of_parts,             │
│    year_released, <theme columns>                   │
│  • Log dataset + params to MLflow                   │
│  • Upsert to feature table (primary key: set_number)│
│  • First run only: write lego_set_features_baseline │
└────────────────────────┬────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────┐
│  Unity Catalog: lego_set_features                    │
│  • set_number: string                               │
│  • number_of_parts: float                           │
│  • year_released: int                               │
│  • Technic: int  (one-hot theme columns)             │
│  • Star_Wars: int                                    │
│  • ...                                               │
│  • Other: int  (residual + source's own Other theme) │
│  • Format: Delta Lake                                │
└────────────────────────┬────────────────────────────┘
                         │ spark.read.table()
                         ▼
┌─────────────────────────────────────────────────────┐
│  STAGE 2: Model Training                             │
│  • Convert to Pandas                                 │
│  • Features: year_released + one-hot theme columns   │
│  • Target: number_of_parts                           │
│  • Train/test split: 80/20                           │
│  • Fit AdaBoostRegressor (n_estimators=50,           │
│    learning_rate=1.0)                                │
│  • Log RMSE, params, datasets, model to MLflow       │
│  • Register model → set "champion" alias             │
└───────────────┬─────────────────┬────────────────────┘
                │                 │
                ▼                 ▼
┌──────────────────────┐  ┌───────────────────────────┐
│  MLflow Registry     │  │  STAGE 3: Model Serving   │
│  • Model: sklearn    │  │  • Deploy champion model  │
│    AdaBoostRegressor │  │    to Databricks endpoint │
│  • Alias: "champion" │  │  • Configure permissions  │
│  • Artifacts:        │  │    and workload size      │
│    model_config.yml  │  └───────────────────────────┘
└──────────┬───────────┘
           │ mlflow.pyfunc.load_model(@champion)
           ▼
┌─────────────────────────────────────────────────────┐
│  STAGE 4: Batch Prediction                           │
│  • Read lego_set_features                            │
│  • Sample random subset (n=10, seed=42)              │
│  • Convert to Pandas, drop set_number + number_of_parts│
│  • Predict piece counts with champion model          │
│  • Add Predicted_number_of_parts column              │
│  • Convert back to Spark                             │
│  • Upsert to predictions table (primary key: set_number)│
└────────────────────────┬────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────┐
│  Unity Catalog: lego_parts_predictor_batch_predictions│
│  • set_number: string                               │
│  • number_of_parts: float                            │
│  • <theme columns>: int                              │
│  • Predicted_number_of_parts: float                  │
│  • Format: Delta Lake                                │
└─────────────────────────────────────────────────────┘
```

## Serving Request Flow

```
CLIENT REQUEST
  │
  │ POST /invocations
  │ Body: {"inputs": [[2005, 1, 0, 0, ...]]}
  │ (unnamed tensor: one row per prediction, values positional —
  │  year_released, then one-hot theme columns, no field names)
  ▼
┌────────────────────────────────────────┐
│  Databricks Model Serving Endpoint     │
│  lego_parts_predictor_endpoint         │
└────────────────────────────────────────┘
  │
  └─► AdaBoostRegressor.predict(features) → Predicted_number_of_parts (float)

RESPONSE
{
  "predictions": [312.0]
}
```

## Data Transformations

### Preprocessing Transformation

```
lego_sets_bronze
  • year_released: "2005.0"  (string)
  • number_of_parts: "312.0"  (string)
  • theme_name: "Technic"
          ↓ drop_invalid_rows() + fix_data_types()
  • year_released: 2005  (int)
  • number_of_parts: 312.0  (float)
          ↓ filter_buildable_sets()  [number_of_parts > 0]
          ↓ deduplicate_on_key()
          ↓ build_category_vocabulary()  [top 100 themes + Other]
          ↓ apply_year_window()  [drift-replay lever]
          ↓ create_category_onehot_encodings()
  • Technic: 1
  • Star_Wars: 0
  • Other: 0
  • ...
          ↓ select(set_number, number_of_parts, year_released, *themes)
lego_set_features
  • set_number, number_of_parts, year_released, <one-hot theme columns>
```

### Training Transformation

```
lego_set_features (Spark DataFrame)
          ↓ toPandas()
Pandas DataFrame
          ↓ drop set_number (id), number_of_parts (target)
X  →  year + theme feature matrix  [n_samples × (1 + n_themes)]
y  →  number_of_parts array         [n_samples]
          ↓ train_test_split(test_size=0.2)
X_train, X_test, y_train, y_test
          ↓ AdaBoostRegressor(n_estimators=50).fit(X_train, y_train)
Trained model  →  RMSE logged to MLflow
```

### Batch Prediction Transformation

```
lego_set_features (Spark DataFrame)
          ↓ pick_random_subset(n=10, seed=42)
10-row Spark DataFrame
          ↓ toPandas() → drop(["set_number", "number_of_parts"])
Feature matrix  [10 × (1 + n_themes)]
          ↓ champion_model.predict()
Predicted_number_of_parts array  [10]
          ↓ spark.createDataFrame()
Spark DataFrame with Predicted_number_of_parts column
          ↓ upsert_delta_table(primary_key="set_number")
lego_parts_predictor_batch_predictions (Delta)
```

---

For detailed architecture, see [architecture.md](architecture.md).
