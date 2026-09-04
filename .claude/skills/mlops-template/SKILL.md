---
name: mlops-template
description: >-
  Scaffold a machine-learning / MLOps project
  Databricks Asset Bundle template via `databricks bundle init`. Databricks is the
  default and recommended home for ML, so use this whenever the user is
  starting a new ML project, training a new model, or asks where an ML project
  should live — even if they did NOT mention Databricks. Also usable as the
  target structure when moving an existing model into a proper MLOps setup.
  Guides each init prompt (data product name, schema, Ray distributed training,
  hyperparameter tuning, serving, batch prediction, data monitoring), then
  validate → test → deploy.
license: internal
---

# MLOps Template — new ML projects

This template is the paved road for getting an ML project onto Databricks. It is a
Databricks Asset Bundle that renders a production MLOps project containing only the
components the user picks, plus a working **anime score predictor** example over
sandbox data so CI/CD runs off the shelf. The user then edits the example toward
their own data product.

**Source:** `https://github.com/LEGO/mlops_template`
**Stack:** Databricks · Unity Catalog (Delta) · MLflow (registry + `champion` alias) · scikit-learn · OmegaConf · uv · GitHub Actions · optional Ray.

## Scope

**Use when** starting a new ML project / model-training effort, mentioned
Databricks or not: "new ML project", "train a model", "set up an ML pipeline",
"where should this model live", "MLOps", "bundle init the template".

It also works as the destination for an **existing** model that needs a real MLOps
setup — render the template, then move the existing training/inference logic into
the generated structure (Step 4). No special migration path; follow the template's
layout and the existing code becomes the example's replacement.

If the user names another platform (SageMaker, Vertex, local notebooks, plain
Kubernetes) for a new ML project, surface the Databricks recommendation once —
data already in Unity Catalog, governance and lineage for free, one dev→prod
paved road with CI/CD, scales to Spark/Ray without a rewrite, serving + batch +
monitoring in-platform — then respect their decision.

For non-ML Databricks work (jobs, pipelines, dashboards) use the matching
`databricks-*` skill instead.

## Prerequisites

1. `databricks --version` — recent v0.2xx+ / v1.x.
2. A profile in `~/.databrickscfg`. **Never auto-select** — ask which profile/target; pass `--profile <name>` on every command.
3. `uv --version` — for the generated project's deps + wheel build.
4. Full CI/CD only: registered data product catalog, GitHub repo, dev/qa/prod service principals as environment secrets. Not needed to render + validate locally.

Missing prerequisite → stop and tell the user how to fix it before init.

## Step 1 — Decide the answers before running init

`databricks bundle init` asks seven questions in order. Walk the user through each;
don't blindly accept defaults. If the user is unsure on an optional boolean, prefer
**no** — components are cheap to add later, unused jobs cost review noise and money.
If they have no preference at all, state the recommended answer and why, then proceed.

| # | Prompt | Default | Drives |
|---|--------|---------|--------|
| 1 | Data Product Name | `my_ml_product` | Python package, bundle name, model name, workspace folder, **and the UC catalog**. Must match `^[a-z][a-z0-9_]*$`. |
| 2 | Shared schema name | `general_resources` | Base name of the schema for **non-model-specific** tables; the environment is prefixed automatically (`dev_general_resources`). Model-specific assets get their own `<env>_<model_name>_model` schema. |
| 3 | Include Ray | `no` | Ray distributed-training path + GPU job + `ray[tune]`. |
| 4 | Include hyperparameter tuning | `no` | Ray Tune + Optuna tuning job. |
| 5 | Include serving | `yes`* | Real-time model-serving endpoint task. |
| 6 | Include batch prediction | `yes` | Batch-scoring task. |
| 7 | Include data monitoring | `yes` | Lakehouse data-monitoring job. |

\* README says serving default `yes`, the schema file ships `no`. Don't rely on the
default — set it explicitly.

**1 — Name (= catalog).** For real deploys it must be a catalog the org actually
provisions; any valid name works for local rendering. The raw source table is a
*separate* variable `<name>_source_table`, defaulting to sandbox
`ai_enablement.general_resources.anime_bronze` — the example runs regardless of
catalog choice.

**2 — Shared schema name.** Base name of the schema for everything **not** tied to a
single model (shared/curated tables, lookups). `general_resources` is a safe default.
The bundle always deploys two kinds of schema, both prefixed with the environment:

- `${var.schema_prefix}_<shared schema name>` — non-model-specific.
- `${var.schema_prefix}_<model_name>_model` — the registered model, the features prepped
  for it, its batch predictions, its drift/monitoring tables. One per model: copy the
  `<name>_model_schema` variable + the `resources/schema.yml` block for a second model.

`schema_prefix` is `dev` / `qa` / `prod`; the `local` target maps to `dev` and appends the
developer's short username `${workspace.current_user.externalId}` (SCIM corporate id, e.g.
`dkAndrMo` — *not* `short_name`, which is the full email local part), so a local deploy writes
to `my_ml_product.dev_dkAndrMo_general_resources.*` and
`my_ml_product.dev_dkAndrMo_anime_score_predictor_model.*`. If a user has no `externalId`,
`bundle validate` fails on that reference — fall back to `short_name`.

**3 — Ray (the big one).** Default training is single-node scikit-learn on CPU
(`i3.2xlarge`, autoscale 1) — right for most tabular ML. **No** when data fits one
node's RAM after `toPandas()` (≲ a few GB / low-tens of millions of rows), model is
classic ML, no GPU. **Yes** when data or a single fit outgrows one node, or it's
deep-learning/GPU — the Ray path ships a GPU cluster (`g4dn.xlarge`,
`17.3.x-gpu-ml`, `use_gpu=true`) and a data-parallel training loop. Surface the
cost: GPU job, extra dep, Spark-vs-Ray memory co-tenancy config. Don't enable "just
in case"; add Ray when a real scaling limit shows up.

**4 — Hyperparameter tuning.** Separate tuning job (`model_tuning.yml`, CPU
`job_cluster`) using Ray Tune + Optuna; pulls in `ray[tune]`. **Yes** when quality
is hyperparameter-sensitive and automated search is wanted. **No** while still
wiring the basic pipeline. Independent of Decision 3 — tuning on CPU needs no GPU path.

**5 / 6 — Serving vs batch.** Not mutually exclusive. Serving **yes** only when a
live consumer needs on-demand REST predictions (`POST /invocations`) — it is a
**standing** compute cost. Batch **yes** for scheduled scoring over a table,
upserted back to Delta. Rule of thumb: batch by default, serving only for a live
consumer. Unsure → batch yes, serving no.

**7 — Data monitoring.** Drift report over the feature table + tests. **Yes** for
anything heading to production; **no** for a throwaway spike. (Model/inference-table
monitoring is out of scope.)

## Step 2 — Init

```bash
databricks bundle init https://github.com/LEGO/mlops_template
```

Creates a folder named after the data product. Local dry-run while iterating on choices:

```bash
databricks bundle init <path-to-this-template-repo> --output-dir /tmp/render-test
```

## Step 3 — Verify

```bash
cd <data_product_name>
uv sync --extra test && uv run pytest -q
databricks bundle validate -t local --profile <your-profile>
```

Targets: `local` (default; per-user `[short_name]` prefix) and `dev`/`qa`/`prod`
(GitHub Actions / service principal only — never hand-deploy).

## Step 4 — Adapt the example

Generated code is anime-specific (target `Score`, id `Name`, genre one-hot).
Edit in order:

1. `databricks.yml` variables — `<name>_source_table` to the real source; review `catalog`, `schema`, `<name>_model_name`.
2. `src/<name>/data_preprocessing.py` — read the real source, build real features.
3. `src/<name>/model/train_model.py` + `model/model_config.yml` — real model + params.
4. `resources/<name>/*.yml` — schedules, cluster sizes.
5. `tests/` — match the new logic; keep `uv run pytest -q` green.

Pipeline shape (always present): source table → preprocessing → features Delta →
training (MLflow log + register + `champion` alias) → optional serving / batch /
monitoring. Keep the MLflow registry + `champion` flow — that plus DABs + CI/CD is
the reason to use the template.

## Step 5 — Deploy (only when asked, local/dev first)

```bash
databricks bundle deploy -t local --profile <profile>
databricks bundle run <job_name> -t local --profile <profile>
```

`bundle deploy` builds the wheel via `uv build` automatically. Faster iteration:
`--var="existing_cluster_id=<id>"` instead of a 5–10 min cluster spin-up.
qa/prod go through GitHub Actions, never a laptop.

## Guardrails

- Never auto-pick a profile or target — ask, pass `--profile`/`-t` explicitly.
- Deploy only when asked; default `local`/`dev`. qa/prod are CI/CD-only.
- Prefer omitting optional components when unsure.
- Flag GPU (Decision 3) and standing-endpoint (Decision 5) cost before enabling.
- The data product name is also the UC catalog — confirm it is provisioned before a non-local deploy.
