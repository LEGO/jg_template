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
| **Schema** | `general_resources` | Unity Catalog output schema. |
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

The generated project contains a working **anime score predictor** example over sandbox data, so
CI/CD runs immediately; you then edit the example to fit your own data. See the generated project's
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
