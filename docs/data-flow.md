# Data Flow Documentation

This document provides a high-level visualization of data flow through the pipeline.

## End-to-End Data Flow

```
┌─────────────────────────────────────────────────────┐
│  Unity Catalog: anime_bronze                        │
│  • MAL_ID: string                                   │
│  • Name: string                                     │
│  • Score: string                                    │
│  • Genres: string  (comma-separated)                │
│  • Synopsis: string                                 │
└────────────────────────┬────────────────────────────┘
                         │ spark.table()
                         ▼
┌─────────────────────────────────────────────────────┐
│  STAGE 1: Data Preprocessing                        │
│  • Cast Score string → float                        │
│  • Split Genres → one-hot encoded columns           │
│  • Select: Name, Score, <genre columns>             │
│  • Log dataset + params to MLflow                   │
│  • Upsert to feature table (primary key: Name)      │
└────────────────────────┬────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────┐
│  Unity Catalog: anime_features                      │
│  • Name: string                                     │
│  • Score: float                                     │
│  • Action: int  (one-hot genre columns)             │
│  • Comedy: int                                      │
│  • ...                                              │
│  • Sci-Fi: int                                      │
│  • Format: Delta Lake                               │
└────────────────────────┬────────────────────────────┘
                         │ spark.read.table()
                         ▼
┌─────────────────────────────────────────────────────┐
│  STAGE 2: Model Training                            │
│  • Convert to Pandas                                │
│  • Features: all columns except Name, Score         │
│  • Target: Score                                    │
│  • Train/test split: 80/20                          │
│  • Fit Lasso regression (alpha=1.0, max_iter=1000)  │
│  • Log RMSE, params, datasets, model to MLflow      │
│  • Register model → set "champion" alias            │
└───────────────┬─────────────────┬───────────────────┘
                │                 │
                ▼                 ▼
┌──────────────────────┐  ┌───────────────────────────┐
│  MLflow Registry     │  │  STAGE 3: Model Serving   │
│  • Model: sklearn    │  │  • Deploy champion model  │
│    Lasso             │  │    to Databricks endpoint │
│  • Alias: "champion" │  │  • Configure permissions  │
│  • Artifacts:        │  │    and workload size      │
│    model_config.yml  │  └───────────────────────────┘
└──────────┬───────────┘
           │ mlflow.pyfunc.load_model(@champion)
           ▼
┌─────────────────────────────────────────────────────┐
│  STAGE 4: Batch Prediction                          │
│  • Read anime_features                              │
│  • Sample random subset (n=10, seed=42)             │
│  • Convert to Pandas, drop Name + Score             │
│  • Predict scores with champion model               │
│  • Add Predicted_Score column                       │
│  • Convert back to Spark                            │
│  • Upsert to predictions table (primary key: Name)  │
└────────────────────────┬────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────┐
│  Unity Catalog: anime_score_predictor_batch_predictions │
│  • Name: string                                     │
│  • Score: float                                     │
│  • <genre columns>: int                             │
│  • Predicted_Score: float                           │
│  • Format: Delta Lake                               │
└─────────────────────────────────────────────────────┘
```

## Serving Request Flow

```
CLIENT REQUEST
  │
  │ POST /invocations
  │ Body: {"dataframe_records": [{"Action": 1, "Comedy": 0, ...}]}
  ▼
┌────────────────────────────────────────┐
│  Databricks Model Serving Endpoint     │
│  anime_score_predictor_endpoint        │
└────────────────────────────────────────┘
  │
  └─► Lasso.predict(features) → Predicted_Score (float)

RESPONSE
{
  "predictions": [7.42]
}
```

## Data Transformations

### Preprocessing Transformation

```
anime_bronze
  • Score: "7.42"  (string)
  • Genres: "Action, Comedy, Sci-Fi"
          ↓ fix_data_types()
  • Score: 7.42  (float)
          ↓ create_genre_onehot_encodings()
  • Action: 1
  • Comedy: 1
  • Sci-Fi: 1
  • Romance: 0
  • ...
          ↓ select(Name, Score, *genres)
anime_features
  • Name, Score, <one-hot genre columns>
```

### Training Transformation

```
anime_features (Spark DataFrame)
          ↓ toPandas()
Pandas DataFrame
          ↓ drop Name (id), Score (target)
X  →  genre feature matrix  [n_samples × n_genres]
y  →  Score array           [n_samples]
          ↓ train_test_split(test_size=0.2)
X_train, X_test, y_train, y_test
          ↓ Lasso(alpha=1.0).fit(X_train, y_train)
Trained model  →  RMSE logged to MLflow
```

### Batch Prediction Transformation

```
anime_features (Spark DataFrame)
          ↓ pick_random_subset(n=10, seed=42)
10-row Spark DataFrame
          ↓ toPandas() → drop(["Name", "Score"])
Feature matrix  [10 × n_genres]
          ↓ champion_model.predict()
Predicted_Score array  [10]
          ↓ spark.createDataFrame()
Spark DataFrame with Predicted_Score column
          ↓ upsert_delta_table(primary_key="Name")
anime_score_predictor_batch_predictions (Delta)
```

---

For detailed architecture, see [architecture.md](architecture.md).
