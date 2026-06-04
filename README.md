![Databricks](https://img.shields.io/badge/Databricks-D14836?style=for-the-badge&logo=databricks&logoColor=white)
![Python](https://img.shields.io/badge/PYTHON-3776AB?style=for-the-badge&logo=python&logoColor=ffdd54)

# mlops-template

An MLOps template for deploying and serving machine learning models on Databricks with production-ready CI/CD integration.

## Overview

The template uses a simple sandbox data set to show how one can build and deploy a machine learning model into NEXUS. Using sandbox data that is accessible through the AI Enablement catalog we can ensure that the template works off the shelf with little changes needed to make the CI/CD part run for any product team. 

### Default workflows

The orchestration in this template is set up with three workflows. The chosen separation allows different scheduling and ensures that each part of the pipeline is decoupled. 

1. **Preprocessing/Feature engineering** - A simple preprocessing example 
2. **Hyperparameter tuning** - A simple hyperparameter tuning example with RAY
3. **Model training** - One version that trains the model on a single cluster and a Ray version which distributes the training.
4. **inference** - Both batch and live inference examples

The repo follows best practices and recommendations from [MLOps maturity framework](https://baseplate.legogroup.io/catalog/default/component/ds_ai_handbook/docs/traditional_ml/docs/maturity_levels/mlops-maturity-level-checklist/) and it points to each component in the **Maturity Level Framekwork** and explain how each script comply with the levels. 

Orchestration for your project might look different. We advice to follow the principles of the [AI Handbook](https://baseplate.legogroup.io/catalog/default/component/ds_ai_handbook/docs) when rewriting the template to fit your data product. 

## Project structure

This is a brief outline of the most important files in the project.

```text
.
├── resources/
│   ├── data_ingestion.yml           # Databricks bundle job for data ingestion
│   ├── model_training_ray.yml       # Databricks bundle job for model training
│   ├── model_training.yml           # Databricks bundle job for model training
│   ├── model_tuning.yml             # Databricks bundle job for model tuning
├── src/
│   ├── common/                      # Shared utilities used across pipeline steps
│   │   ├── mlflow_helper.py         # Helper functions for tracking & registering models
│   │   ├── spark_helper.py          # Spark session and Delta table helpers
│   │   ├── ray_helper.py            # Spark session and Delta table helpers
│   │   └── utils.py                 # General utility functions and logging
│   └── anime_score_predictor/       # Python scripts run as wheels for different pipeline steps
│       ├── batch_prediction.py      # Entry point for batch inference
│       ├── data_preprocessing.py    # Entry point for data preprocessing
│       └── model/
│           ├── model_config.yml     # Model configurations
│           ├── serve.py             # Entry point for endpoint / model serving tests
│           ├── train_model_ray.py   # Entry point for model training using ray
│           ├── train_model.py       # Entry point for model training
│           └── tune_model_ray.py    # Entry point for model tuning
├── tests/
│   └── test_sample.py               # Sample unit tests
├── docs/
│   ├── architecture.md              # System architecture and component details
│   ├── data-flow.md                 # Data flow diagrams and pipeline visualizations
│   └── developer-guide.md           # Code examples, debugging tips, and recipes
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

## Getting started 

The repository works off the shelve. However, to fit to your data product one must change standard scripts and redefine artifacts in the automation bundle (DAB)
- [data preparation](./src/anime_score_predictor/data_preprocessing.py) (modify)
- [training](./src/anime_score_predictor/model/train_model.py) (modify)
- [hyperparameter tuning](./src/anime_score_predictor/model/tune_model_ray.py) (modify or delete)
- [batch prediction](./src/anime_score_predictor/batch_prediction.py) (modify or delete)
- [serving](./src/anime_score_predictor/model/serve.py) (modify or delete)
- [bundle variables](databricks.yml) (specific workflow variables found in `resources/*` folder)

Any Machine Learning pipeline needs data preparation, training and prediction scripts to work. This template covers both real-time serving and batch prediction. Also, one can make use of the hyperparameter tuning if needed. 

### Prerequisites

**Local** 
- Python (version as specified in `pyproject.toml`)
- [uv](https://docs.astral.sh/uv/) installed
- [Databricks CLI v0.205+](https://docs.databricks.com/en/dev-tools/cli/index.html) configured (`databricks configure` or OAuth)

**CI/CD**
- Data product ([register here](https://baseplate.legogroup.io/catalog/default/tool/onex))
    - Databricks catalog on NEXUS
    - Github project 
    and environments dev, qa and prod with service principal 
    - Service Principal IDs set as environment secrets in github (dev, qa & prod)

### Deployment

The deployment is symmetrical on each environment and is tied to your branches. The default triggers are
- `dev/**` branches deploys to the `dev` environment. 
- pull requests deploys to `qa` envrionment. 
- `release/*` branches deploys to `prod` environment. 

### Local development

```bash
git clone https://github.com/LEGO/mlops_template.git
cd mlops_template
```

### Install dependencies locally

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
- **Ray Documentation**: [https://docs.ray.io/en/latest/index.html#](https://docs.ray.io/en/latest/index.html#)
