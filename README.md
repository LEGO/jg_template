![Databricks](https://img.shields.io/badge/Databricks-D14836?style=for-the-badge&logo=databricks&logoColor=white)

# LEGO MLOps Template

A [Databricks Asset Bundle template](https://docs.databricks.com/dev-tools/bundles/templates.html)
that scaffolds an MLOps project on Databricks with only the components you need.

## Usage

```bash
databricks bundle init https://github.com/LEGO/mlops_template
```

You'll be prompted for a few values, then a ready-to-run project is generated in a folder named
after your data product.

## Prompts

| Prompt | Default | Purpose |
|---|---|---|
| **Data Product Name** | `my_ml_product` | Python package, bundle name, model name, resource identifiers, **and the UC catalog** (catalog = this name). Must be a valid Python/UC name (`^[a-z][a-z0-9_]*$`). |
| **Shared schema name** | `general_resources` | Base name of the schema for non-model-specific tables. Prefixed with the environment (`dev_general_resources`, …). Model-specific assets land in a separate `<env>_<model_name>_model` schema. |
| **Include Ray** | `no` | Ray distributed-training path + GPU job + `ray[tune]` dep. |
| **Include hyperparameter tuning** | `no` | Ray Tune + Optuna tuning job. |
| **Include serving** | `yes` | Real-time model-serving endpoint task. |
| **Include batch prediction** | `yes` | Batch-scoring task. |
| **Include data monitoring** | `yes` | Databricks Lakehouse data-monitoring job. |

With **serving** and **data monitoring** both enabled you also get a prediction drift monitoring
job: it flattens the endpoint's AI Gateway inference payload table and attaches a Lakehouse
`InferenceLog` monitor to the result.

## What you always get (foundational)

Data preparation, model training, model evaluation + registry (MLflow), continuous integration
(tests + workflow), and continuous delivery (the bundle + GitHub Actions). Optional components
above are included or omitted based on your answers — omitted components leave no files behind.

The generated project contains a working **LEGO set-size predictor** example (predicting
a set's piece count from its theme and release year) over sandbox data, so CI/CD runs
immediately; you then edit the example to fit your own data. See the generated project's
own `README.md`.

## Template layout

```text
databricks_template_schema.json      # the prompts
template/
  {{.data_product_name}}/            # rendered into your project folder
    databricks.yml.tmpl              # carries the {{skip}} rules for optional files
    pyproject.toml.tmpl
    resources/ src/ tests/ .github/
```

## Developing this template

Render locally without Git to test changes:

```bash
databricks bundle init ./ --output-dir /tmp/render-test
```

## MLOps Maturity Level 2 Compliance

This repository is designed and structured to achieve the highest maturity ratings according to the [LEGO AI Handbook MLOps Maturity Framework](https://baseplate.legogroup.io/docs/default/component/ds_ai_handbook/traditional_ml/docs/maturity_levels/mlops-maturity-level-checklist/):

- **Feature Store (Level 2)**: Modular feature components in `src/feature_store/`, standalone table definitions (`feature_table_config.yml`), dedicated extraction pipelines (`feature_pipeline.py`), and live/online feature serving configuration (`online_serving.py`) utilizing `databricks-feature-engineering` and `feast`.
- **Model Registry (Level 2)**: Centralized MLflow Model Registry integration in `src/common/model_registry.py` with explicit `mlflow.register_model` registration, version tracking, metadata tags, metric-threshold-based promotion gates (`@champion` vs `@challenger`), approval workflows, and rollback capabilities.
- **Continuous Delivery (Level 2)**: Fully automated deployment pipelines via GitHub Actions (`.github/workflows/ci_cd.yml` and `deploy.yml`) with environment symmetry (`dev`, `qa`, `prod`), scheduled retraining triggers (cron), and Infrastructure as Code (`terraform/`) provisioning Databricks workspaces, clusters, schemas, and scheduled jobs.
- **Continuous Integration (Level 2)**: Automated linting (`ruff`), type checking (`mypy`), unit and component integration tests (`pytest`) with coverage reporting executed in CI/CD.
- **Data & Model Monitoring (Level 2)**: Automated Databricks Lakehouse Monitoring with statistical drift detection (`drift_metrics` and baseline comparison), payload unpacking from AI Gateway inference tables (`unpack_inference_table.py`), and automated threshold alerting.

