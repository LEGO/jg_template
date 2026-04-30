![Databricks](https://img.shields.io/badge/Databricks-D14836?style=for-the-badge&logo=databricks&logoColor=white)
![Python](https://img.shields.io/badge/PYTHON-3776AB?style=for-the-badge&logo=python&logoColor=ffdd54)

# mlops-template

An MLOps template for deploying and serving machine learning models on Databricks with production-ready CI/CD integration.

## Overview




### Default workflows

The orchestration in this template is set up with three workflows

1. **Preprocessing/Feature engineering** - A simple preprocessing example
2. **Hyperparameter tuning** - A simple hyperparameter tuning example with RAY
3. **Model training** and **inference** - A simple example of model that trains and serves (batch and on an endpoint). 

This is a working example with a sandbox dataset that follows best practices and recommendations from [MLOps maturity framework](https://baseplate.legogroup.io/catalog/default/component/ds_ai_handbook/docs/traditional_ml/docs/maturity_levels/mlops-maturity-level-checklist/). 

Orchestration for your project might look different. We advice to follow the principles of the [AI Handbook](https://baseplate.legogroup.io/catalog/default/component/ds_ai_handbook/docs).

## Project structure

This is a brief outline of the most important files in the project.

```text
.
├── resources/
│   ├── data_ingestion.yml           # Databricks bundle job for data ingestion
│   ├── model_training.yml           # Databricks bundle job for model training
├── src/
│   ├── common/                      # Shared utilities used across pipeline steps
│   │   ├── mlflow_helper.py         # Helper functions for tracking & registering models
│   │   ├── spark_helper.py          # Spark session and Delta table helpers
│   │   └── utils.py                 # General utility functions and logging
│   └── mlops_pipeline/              # Python scripts run as wheels for different pipeline steps
│       ├── batch_prediction.py      # Entry point for batch inference
│       ├── data_preprocessing.py    # Entry point for data preprocessing
│       ├── hyperparameter_tuning.py # Entry point for hyperparameter-tuning
│       └── model/
│           ├── model_config.yml     # Model configurations
│           ├── serve.py             # Entry point for endpoint / model serving tests
│           └── train_model.py       # Entry point for model training
├── tests/
│   └── test_sample.py               # Sample unit tests
├── docs/
│   ├── architecture.md              # System architecture and component details
│   ├── data-flow.md                 # Data flow diagrams and pipeline visualizations
│   └── developer-guide.md          # Code examples, debugging tips, and recipes
├── databricks.yml                   # Databricks bundle definition (jobs, clusters, permissions)
├── pyproject.toml                   # Python package + uv configuration
└── README.md                        # Docs
```

## Documentation

For detailed technical documentation, see:

- **[Architecture Documentation](docs/architecture.md)** – Comprehensive system architecture, component details, model architecture, deployment architecture, and technology stack
- **[Data Flow & Pipeline Visualizations](docs/data-flow.md)** – Detailed data flow diagrams, pipeline stage breakdowns, and transformation visualizations
- **[Developer Guide](docs/developer-guide.md)** – Code examples, debugging tips, common tasks, and MLflow/Unity Catalog recipes

## Key Databricks Concepts

- **Databricks Bundle** ([`databricks.yml`](databricks.yml)) - declarative IaC for jobs, clusters, and resources
- **Unity Catalog** - data governance with Delta tables and Volumes for data and model features
- **MLflow Registry** - model versioning with aliases ("champion", "challenger")
- **Model Serving** - REST API endpoint for anime score prediction
- **CPU Clusters** - Required for model training (17.3.x ML CPU runtime)

For detailed deployment architecture, see [docs/architecture.md](docs/architecture.md).

## Local development

### Clone the repository

```bash
git clone https://github.com/LEGO/mlops_template.git
cd mlops_template
```

### Prerequisites

- Python (version as specified in `pyproject.toml`)
- [uv](https://docs.astral.sh/uv/) installed
- [Databricks CLI v0.205+](https://docs.databricks.com/en/dev-tools/cli/index.html) configured (`databricks configure` or OAuth)

### Install dependencies

From the repository root:

```bash
uv sync --all-extras
```

## Pipeline deployment (manual via CLI)

To deploy the pipeline manually from your local machine, from the repository root run:

```bash
databricks bundle deploy --target dev --profile dev
```

To deploy to a different environment, change the bundle target and profile accordingly (e.g. `qa`, `prod`).

## Running the pipelines (manual via CLI)

There are several separable pipelines that can be run in the project. To run a pipeline, from the repository root use:

```bash
databricks bundle run --target <ENV> --profile <ENV>
```
and then choose the job you want to run.
Or use databricks connect

## Testing the serving endpoint

For serving endpoint examples, see [docs/developer-guide.md](docs/developer-guide.md#test-serving-endpoint-via-rest-api).

## CI/CD: GitHub Actions bundle deploy-and-run

This repository also contains a GitHub Actions setup that can deploy the Databricks bundle and run a specific job in a given environment. It is implemented as a reusable composite action in `/.github/actions/run/action.yml`.

### Composite action interface

Inputs:

- `environment` – Databricks bundle target to use (one of `dev`, `qa`, `prod`, etc. as defined in `databricks.yml`).
- `databricks_client_id` – OAuth client ID used by the Databricks CLI in CI.
- `databricks_client_secret` – OAuth client secret used by the Databricks CLI in CI.
- `databricks_job` – Name of the job to run (must match a job defined in the bundle resources, for example `model_training_job`).

The action performs two steps from the repository root (where `databricks.yml` lives):

1. `databricks bundle deploy --target <environment>` – deploys the bundle to the target workspace.

### Example workflow

A minimal workflow that uses this composite action was shown in `.github/workflows/bundle-deploy-and-run.yml`.

# Known issues
 - Argparse is done seperately for each job. It could be modularized and parametrized to prevent code duplication.

## Getting Help

- **Architecture details**: See [docs/architecture.md](docs/architecture.md)
- **Data flow diagrams**: See [docs/data-flow.md](docs/data-flow.md)
- **Databricks documentation**: [https://docs.databricks.com](https://docs.databricks.com)
- **MLflow documentation**: [https://mlflow.org/docs](https://mlflow.org/docs)
