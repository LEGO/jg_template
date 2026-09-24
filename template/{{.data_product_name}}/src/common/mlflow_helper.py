import os
from typing import Any, Dict, Optional

import mlflow
from databricks.sdk import WorkspaceClient

from common.utils import get_logger

logger = get_logger()

# Provenance tag applied to every MLflow run, registered model and model version
# produced by a project generated from the MLOps template. Lets you find all
# template-derived models with e.g.
#   mlflow.search_model_versions("tags.source_template = 'mlops_template'")
TEMPLATE_TAG_KEY = "source_template"
TEMPLATE_TAG_VALUE = "mlops_template"

''' 
Model Registry is compliant with level 2 of https://baseplate.legogroup.io/catalog/default/component/ds_ai_handbook/docs/traditional_ml/docs/maturity_levels/7-mlops-model-registry/#model-registry.
Model Evaluation is compliant with level 2 of https://baseplate.legogroup.io/catalog/default/component/ds_ai_handbook/docs/traditional_ml/docs/maturity_levels/3-mlops-model-evaluation/#model-evaluation. 
Model Monitoring is compliant with level 2 of https://baseplate.legogroup.io/catalog/default/component/ds_ai_handbook/docs/traditional_ml/docs/maturity_levels/5-mlops-model-monitoring/#model-monitoring.
Feature Store is compliant with level 2 of https://baseplate.legogroup.io/catalog/default/component/ds_ai_handbook/docs/traditional_ml/docs/maturity_levels/8-mlops-feature-store/#feature-store. 

Explanation: 
Model Evaluation 1) Model evaluation is closely connected with logging experiments and metrics properly. Having logged metrics we are set up for creating a sophisticated model promotion strategy that allows us to evaluate new runs against old deployments automatically using the mlflow api. 
However, in this example we use the newest data to train on but ensure best performance at all times by continously creating a model hyperparameter tuning that feeds parameters to the main training job. 
Model Evaluation depends on the use case, e.g. for time-series problems it is adviced to always use the most recent data but tune input hyperparameters to ensure best performance, for other use cases you might want to evaluate on a fixed test set and compare the performance of the new model to the current champion model. This should be reflected in the model promotion strategy, i.e. champion labeling of new models. 

Model Monitoring 1) Statistics about the model which will go into calculating the model performance should automatically be captured via mlflow metric logging. 
What statistics to log is highly depending on the use case and model choice.  
'''


def _ensure_experiment_directory_exists(experiment_path: str) -> None:
    """Creates parent directories for the experiment path if they don't exist.

    Args:
        experiment_path (str): The full path to the MLflow experiment.
    """
    parent_dir = os.path.dirname(experiment_path)
    if not parent_dir:
        return

    try:
        ws = WorkspaceClient()
        ws.workspace.mkdirs(parent_dir)
        logger.info(f"Created directory: {parent_dir}")
    except Exception as e:
        # Directory might already exist, which is fine
        logger.debug(f"Directory creation note: {e}")


def start_mlflow_experiment_and_run(experiment_path: str, artifact_location: str | None = None) -> mlflow.ActiveRun:
    """Starts an MLflow experiment and initiates a run.

    Checks if the experiment exists, creates it if not (ensuring the parent
    directory exists), sets the experiment, and starts a new run.
    Any existing active run is ended before starting a new one.

    Args:
        experiment_path (str): The name/path of the MLflow experiment.
        artifact_location (str | None): Optional artifact storage location (e.g., Unity Catalog Volume path).
            If None, uses default Databricks location.

    Returns:
        mlflow.ActiveRun: The active MLflow run context.
    """
    mlflow.set_registry_uri("databricks-uc")

    if mlflow.get_experiment_by_name(experiment_path) is None:
        _ensure_experiment_directory_exists(experiment_path)
        if artifact_location:
            mlflow.create_experiment(name=experiment_path, artifact_location=artifact_location)
        else:
            mlflow.create_experiment(name=experiment_path)
        mlflow.set_experiment(experiment_path)
    else:
        mlflow.set_experiment(experiment_path)

    try:
        mlflow.end_run()
    except Exception as e:
        logger.info(e)

    run = mlflow.start_run()
    mlflow.set_tag(TEMPLATE_TAG_KEY, TEMPLATE_TAG_VALUE)
    return run


def register_model_in_registry(
    model_uri: str,
    fully_qualified_model_name: str,
    tags: dict | None = None,
) -> Any:
    """Registers a model to the MLflow Model Registry (Level 1 & 2 MLOps compliance).

    Explicitly calls mlflow.register_model to track model version, metadata, and lineage
    in the centralized model registry.
    """
    logger.info(f"Registering model from {model_uri} to registry {fully_qualified_model_name}")
    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=fully_qualified_model_name,
    )
    tag_model_as_template_generated(fully_qualified_model_name, version=registered_model.version)
    return registered_model


def tag_model_as_template_generated(
    fully_qualified_model_name: str, version: str | int | None = None
) -> None:
    """Tags a registered model (and one of its versions) as template-generated.

    Applies TEMPLATE_TAG_KEY=TEMPLATE_TAG_VALUE to the registered model and, when a
    version is given, to that model version. Tagging failures are logged but never
    fail the training job.

    Args:
        fully_qualified_model_name (str): Unity Catalog model name ("catalog.schema.model_name").
        version (str | int | None): Model version to tag. If None, only the registered model is tagged.
    """
    mlflow_client = mlflow.MlflowClient()

    try:
        mlflow_client.set_registered_model_tag(
            name=fully_qualified_model_name,
            key=TEMPLATE_TAG_KEY,
            value=TEMPLATE_TAG_VALUE,
        )
        if version is not None:
            mlflow_client.set_model_version_tag(
                name=fully_qualified_model_name,
                version=str(version),
                key=TEMPLATE_TAG_KEY,
                value=TEMPLATE_TAG_VALUE,
            )
    except Exception as e:
        logger.warning(f"Could not set template provenance tag on {fully_qualified_model_name}: {e}")


def set_champion_alias_on_logged_model(
    fully_qualified_model_name: str, model_alias: str = "champion"
) -> None:
    '''Sets the specified alias on the latest version of the registered model.
    
    Args:
        fully_qualified_model_name (str): The fully qualified name of the model in the Unity Catalog (e.g., "catalog.schema.model_name").
        model_alias (str, optional): The alias to assign to the latest model version. Defaults to "champion".
        
    '''
    mlflow_client = mlflow.MlflowClient()
    
    #NOTE here you can introduce a more sophisticated lgic to determine which model version to promote, e.g. based on a specific metric or tag, rather than just taking the latest version
    versions = mlflow_client.search_model_versions(
        filter_string=f"name='{fully_qualified_model_name}'",
    )
    latest_version = max(versions, key=lambda v: int(v.version))

    mlflow_client.set_registered_model_alias(
        name=fully_qualified_model_name,
        alias=model_alias,
        version=latest_version.version,
    )

    tag_model_as_template_generated(fully_qualified_model_name, version=latest_version.version)


def get_metric_from_model_alias(
    model_name: str,
    alias: str = "champion",
    metric_name: str = "rmse",
    client: Optional[mlflow.MlflowClient] = None,
) -> Optional[float]:
    """Retrieves a metric value from a registered model version identified by alias.

    Args:
        model_name: Fully qualified model name in MLflow Model Registry.
        alias: Model alias (e.g., 'champion', 'challenger').
        metric_name: Name of the metric to look up.
        client: Optional MLflowClient instance.

    Returns:
        Optional[float]: Metric value or None if not found.
    """
    client = client or mlflow.MlflowClient()
    try:
        model_version = client.get_model_version_by_alias(name=model_name, alias=alias)
        run = client.get_run(model_version.run_id)
        if metric_name in run.data.metrics:
            return float(run.data.metrics[metric_name])
        # Fallback to tag if logged as tag
        if metric_name in model_version.tags:
            return float(model_version.tags[metric_name])
        return None
    except Exception as e:
        logger.warning(f"Could not retrieve {metric_name} for {model_name}@{alias}: {e}")
        return None
