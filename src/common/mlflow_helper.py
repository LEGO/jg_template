import os

import mlflow
from databricks.sdk import WorkspaceClient

from common.utils import get_logger

logger = get_logger()

''' 
Model Evaluation is (arguably) compliant with level 2 of https://baseplate.legogroup.io/catalog/default/component/ds_ai_handbook/docs/traditional_ml/docs/maturity_levels/3-mlops-model-evaluation/#model-evaluation. 
Model Monitoring is (arguably) compliant with level 2 of https://baseplate.legogroup.io/catalog/default/component/ds_ai_handbook/docs/traditional_ml/docs/maturity_levels/5-mlops-model-monitoring/#model-monitoring. 

Explanation: 
ME1) Model evaluation is closely connected with logging experiments and metrics properly. Having logged metrics we are set up for creating a sophisticated model promotion strategy that allows us to evaluate new runs against old deployments automatically using the mlflow api. 
However, in this example we use the newest data to train on but ensure best performance at all times by continously creating a model hyperparameter tuning that feeds parameters to the main training job. 
Model Evaluation depends on the use case, e.g. for time-series problems it is adviced to always use the most recent data but tune input hyperparameters to ensure best performance, for other use cases you might want to evaluate on a fixed test set and compare the performance of the new model to the current champion model. This should be reflected in the model promotion strategy, i.e. champion labeling of new models. 

MM1) Statistics about the model which will go into calculating the model performance should automatically be captured via mlflow metric logging. 
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
    return mlflow.start_run()


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
