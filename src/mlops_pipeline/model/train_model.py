import argparse
import mlflow
from mlflow.data.numpy_dataset import from_numpy
from mlflow.models import infer_signature
from sklearn.ensemble import AdaBoostRegressor
import numpy as np
from pyspark.sql import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from omegaconf import OmegaConf
from common.spark_helper import get_spark_session
from common.utils import get_logger
from common.mlflow_helper import start_mlflow_experiment_and_run, set_champion_alias_on_logged_model
from pathlib import Path

logger = get_logger()


def _load_model_config() -> dict:
    '''
    Loads the model configuration from a YAML file.

    Returns:
        dict: The model configuration.

    '''
    config_path = Path(__file__).parent / "model_config.yml"
    return OmegaConf.load(config_path)

def _load_model_config_from_hyperparameter_tuning(experiment_path: str) -> dict:
    '''
    Fetches the best hyperparameters from the most recent tuning run in the given MLflow experiment.

    Tuning runs are identified by the presence of params prefixed with "best_" (as logged by
    tune_model.py and tune_model_ray.py). The latest run by start time is used.

    Args:
        experiment_path: MLflow experiment path, e.g. "/Shared/mlops_pipeline/dev/.../model_training".

    Returns:
        Dict of hyperparameter names to values, with the "best_" prefix stripped.

    Raises:
        ValueError: If no tuning runs with "best_" params are found in the experiment.
    '''
    client = mlflow.MlflowClient()

    experiment = client.get_experiment_by_name(experiment_path)

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
    )

    cfg = _load_model_config()
    type_map: dict = {
        param: spec["type"]
        for param, spec in OmegaConf.to_container(cfg.tuning, resolve=True).items()
        if isinstance(spec, dict) and "type" in spec
    }
    _casters = {"int": int, "float": float, "str": str, "bool": bool}

    for run in runs:
        best_params = {k: v for k, v in run.data.params.items() if k.startswith("best_")}
        if best_params:
            cleaned_params = {
                (name := k.replace("best_", "")): _casters.get(type_map.get(name, "float"), float)(v)
                for k, v in best_params.items()
            }
            logger.info(f"Loaded best hyperparameters from run {run.info.run_id}: {cleaned_params}")
            return cleaned_params

    raise ValueError(
        f"No tuning runs with 'best_' parameters found in experiment: {experiment_path}"
    )


def train_model(
    dataframe: DataFrame,
    fully_qualified_model_name: str,
    experiment_path: str,
    model_alias: str = "champion",
    hyperparameter_experiment_path: str | None = None,
) -> None:
    '''
    Trains an AdaBoost model using the provided DataFrame and logs the model and metrics to MLflow.

    Args:
        dataframe (DataFrame): The input DataFrame containing features and target.
        fully_qualified_model_name (str): The fully qualified name of the model to be registered.
        experiment_path (str): The path of the MLflow experiment.
        model_alias (str, optional): The alias to assign to the trained model. Defaults to "champion".
        hyperparameter_experiment_path (str | None, optional): If provided, the MLflow experiment path to fetch the best hyperparameters from. Defaults to None.
    '''
    cfg = _load_model_config()

    if hyperparameter_experiment_path:
        try: 
            ada_params = _load_model_config_from_hyperparameter_tuning(hyperparameter_experiment_path)
        except Exception as e:
            logger.warning(f"Failed to load hyperparameters from tuning experiment, falling back to defaults. Error: {e}")
            ada_params:dict = OmegaConf.to_container(cfg.default_ada, resolve=True)
    else: 
        ada_params:dict = OmegaConf.to_container(cfg.default_ada, resolve=True)
    column_params:dict = OmegaConf.to_container(cfg.columns, resolve=True)
    
    start_mlflow_experiment_and_run(experiment_path)

    logger.info("Preparing data for training")
    pdf = dataframe.toPandas()

    feature_cols = [c for c in pdf.columns if c not in (column_params.get("id"), column_params.get("target_name"))]
    X = pdf[feature_cols].values
    y = pdf[column_params.get("target_name")].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    logger.info("Logging datasets in mlflow")
    train_dataset = from_numpy(X_train, targets=y_train, name="train")
    test_dataset = from_numpy(X_test, targets=y_test, name="test")

    mlflow.log_input(train_dataset, context="training")
    mlflow.log_input(test_dataset, context="test")

    model = AdaBoostRegressor(**ada_params)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    signature = infer_signature(X_train, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    logger.info(f"RMSE: {rmse:.4f}")

    mlflow.log_metrics({"rmse": rmse})
    mlflow.log_params(ada_params)

    mlflow.log_artifact(str(Path(__file__).parent / "model_config.yml"), artifact_path="model_config")

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        registered_model_name=fully_qualified_model_name,
    )

    set_champion_alias_on_logged_model(fully_qualified_model_name, model_alias=model_alias)
    
    mlflow.end_run()

def _parse_args() -> argparse.Namespace:
    """Parses and returns CLI arguments for the ingestion pipeline.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run the model training pipeline.",
    )
    parser.add_argument(
        "--catalog",
        required=True,
        help="Catalog name where the training table will be written.",
    )
    parser.add_argument(
        "--schema",
        required=True,
        help="Schema (database) name where the training table will be written.",
    )
    parser.add_argument(
        "--feature_store_table_name",
        required=True,
        help="Name of the feature table to be registered.",
    )
    parser.add_argument(
        "--model_name",
        required=True,
        help="Name of the model to be trained.",
    )
    parser.add_argument(
        "--experiment_path",
        required=True,
        help="Path of the experiment to be used for logging.",
    )
    parser.add_argument(
        "--model_alias",
        required=False,
        default="champion",
    )
    parser.add_argument(
        "--hyperparameter_experiment_path",
        required=False,
        default=None,
        help="If provided, the MLflow experiment path to fetch the best hyperparameters from.",
    )
    return parser.parse_args()


def main():
    """Entry point for the model training script.

    Parses arguments, initializes Spark, and runs the model training pipeline.
    """
    logger.info("Starting model training pipeline.")
    args = _parse_args()

    spark = get_spark_session(Path(__file__).stem)

    dataframe = spark.read.table(f"{args.catalog}.{args.schema}.{args.feature_store_table_name}")

    logger.info("Reading preprocessed data from feature store.")
    train_model(
        dataframe=dataframe,
        fully_qualified_model_name=f"{args.catalog}.{args.schema}.{args.model_name}",
        experiment_path=args.experiment_path,
        model_alias=args.model_alias,
        hyperparameter_experiment_path=args.hyperparameter_experiment_path,
    )

if __name__ == "__main__":
    main()
