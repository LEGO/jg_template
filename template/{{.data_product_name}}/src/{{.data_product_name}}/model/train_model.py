import argparse
import mlflow
from mlflow.data.numpy_dataset import from_numpy
from mlflow.models import infer_signature
from sklearn.ensemble import HistGradientBoostingRegressor
import numpy as np
from pyspark.sql import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from omegaconf import OmegaConf
from common.spark_helper import get_spark_session
from common.utils import get_logger, load_model_config
from common.mlflow_helper import start_mlflow_experiment_and_run, set_champion_alias_on_logged_model
from pathlib import Path

logger = get_logger()

config_path = Path(__file__).parent / "model_config.yml"

'''
Model training script compliant with level 2 of https://baseplate.legogroup.io/catalog/default/component/ds_ai_handbook/docs/traditional_ml/docs/maturity_levels/2-mlops-model-training/#model-training.
Model registration is compliant with level 2 of https://baseplate.legogroup.io/catalog/default/component/ds_ai_handbook/docs/traditional_ml/docs/maturity_levels/7-mlops-model-registry/#model-registry.

Explanation: 
Model Training 1) The model training script is designed to be modular and reusable, with clear separation of concerns. Allowing Continuous Training is achieved by scheduling this script to run at regular intervals or in response to specific events, ensuring that the model remains up-to-date with the latest data and features from the feature store.
Model Training 2) The script uses a configuration file (model_config.yml) to manage model parameters and training settings, allowing for easy adjustments without modifying the code. However changes are still anchored in version control.
Model Training 3) The script registers metrics, hyperparameters and data artifacts to an experiment allowing easy tracking and comparison of different training runs.
Model Training 4) This also allows for easy retrieval of the best hyperparameters from tuning experiments, enabling seamless integration between hyperparameter tuning and model training.
Model Training 5) The script reads preprocessed data from the feature store, ensuring that the model training always uses the latest data and features.
Model Registry 1) The model registry is used to govern the launching process of models by reviewing, approving, releasing and rolling back deployments. This is controlled via aliases. 
'''




def _load_model_config_from_hyperparameter_tuning(experiment_path: str) -> dict:
    """
    Fetches the best hyperparameters from the most recent tuning run in the given MLflow experiment.

    Tuning runs are identified by the presence of params prefixed with "best_" (as logged by
    tune_model.py and tune_model_ray.py). The latest run by start time is used.

    Args:
        experiment_path: MLflow experiment path, e.g. "/ai_agency/lego_parts_predictor/dev/.../model_training".

    Returns:
        Dict of hyperparameter names to values, with the "best_" prefix stripped.

    Raises:
        ValueError: If no tuning runs with "best_" params are found in the experiment.
    """
    client = mlflow.MlflowClient()

    experiment = client.get_experiment_by_name(experiment_path)

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
    )

    cfg = load_model_config(config_path=config_path)
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

    raise ValueError(f"No tuning runs with 'best_' parameters found in experiment: {experiment_path}")


def setup_parameters(hyperparameter_experiment_path: str | None, model_params: dict) -> tuple:
    cfg = load_model_config(config_path=config_path)

    if hyperparameter_experiment_path:
        try:
            model_params = _load_model_config_from_hyperparameter_tuning(hyperparameter_experiment_path)
        except Exception as e:
            logger.warning(f"Failed to load hyperparameters from tuning experiment, falling back to defaults. Error: {e}")
            model_params: dict = OmegaConf.to_container(cfg.default_hgb, resolve=True)
    else:
        model_params: dict = OmegaConf.to_container(cfg.default_hgb, resolve=True)
    column_params: dict = OmegaConf.to_container(cfg.columns, resolve=True)

    return column_params, model_params

def prepare_data(dataframe: DataFrame, column_params: dict) -> tuple:
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
    
    return X_train, X_test, y_train, y_test

def train_model(
    dataframe: DataFrame,
    fully_qualified_model_name: str,
    experiment_path: str,
    model_alias: str = "champion",
    hyperparameter_experiment_path: str | None = None,
) -> None:
    """
    Trains a HistGradientBoosting model using the provided DataFrame and logs the model and metrics to MLflow.

    Args:
        dataframe (DataFrame): The input DataFrame containing features and target.
        fully_qualified_model_name (str): The fully qualified name of the model to be registered.
        experiment_path (str): The path of the MLflow experiment.
        model_alias (str, optional): The alias to assign to the trained model. Defaults to "champion".
        hyperparameter_experiment_path (str | None, optional): If provided, the MLflow experiment path to fetch the best hyperparameters from. Defaults to None.
    """
    
    start_mlflow_experiment_and_run(experiment_path)
    
    column_params, model_params = setup_parameters(hyperparameter_experiment_path, model_params={})
    model = HistGradientBoostingRegressor(**model_params)
    
    X_train, X_test, y_train, y_test = prepare_data(dataframe, column_params)

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    signature = infer_signature(X_train, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    baseline_rmse = float(np.sqrt(mean_squared_error(y_test, np.full_like(y_test, y_train.mean(), dtype=float))))
    skill = 1.0 - (rmse / baseline_rmse) if baseline_rmse else 0.0
    logger.info(f"RMSE: {rmse:.4f}  (predict-the-mean baseline = {baseline_rmse:.4f})")
    if rmse >= baseline_rmse:
        logger.warning(
            "Trained model does NOT beat the predict-the-mean baseline "
            f"({rmse:.1f} vs {baseline_rmse:.1f}). The target is heavily right-skewed, so "
            "squared-error learners are dominated by a few very large sets."
        )

    mlflow.log_metrics({"rmse": rmse, "baseline_rmse": baseline_rmse, "skill_vs_baseline": skill})
    mlflow.log_params(model_params)

    mlflow.log_artifact(str(Path(__file__).parent / "model_config.yml"), artifact_path="model_config")

    mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
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
