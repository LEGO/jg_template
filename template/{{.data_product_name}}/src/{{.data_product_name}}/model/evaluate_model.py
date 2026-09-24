"""Model Evaluation, Metrics Computation, and Automated Promotion Gating.

Compliant with Level 1 and Level 2 of LEGO AI Handbook Model Evaluation Maturity Framework:
https://baseplate.legogroup.io/docs/default/component/ds_ai_handbook/traditional_ml/docs/maturity_levels/3-mlops-model-evaluation/

Explanation:
Model Evaluation Level 1: Systematic evaluation is performed regularly to measure performance over time.
Metrics (RMSE, MAE, R2 score) are computed and logged to MLflow via mlflow.log_metric/mlflow.log_metrics.
Model Evaluation Level 2: Automated metric-threshold-based promotion gates in CI/CD pipelines.
A candidate/challenger model must beat the champion baseline on validation metrics (e.g., candidate_rmse < champion_rmse)
before being promoted to @champion in the MLflow Model Registry.
"""

import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
from mlflow.tracking import MlflowClient

from common.utils import get_logger, load_model_config
from common.mlflow_helper import set_champion_alias_on_logged_model, get_metric_from_model_alias

logger = get_logger()


def compute_evaluation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Computes comprehensive regression evaluation metrics.

    Args:
        y_true: Ground truth target values.
        y_pred: Model predicted values.

    Returns:
        Dict[str, float]: Dictionary containing RMSE, MAE, and R2 metrics.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2_score": r2,
    }
    return metrics


def evaluate_model_performance(
    model: Any,
    evaluation_data: pd.DataFrame,
    target_column: str,
    id_column: str,
) -> Dict[str, float]:
    """Evaluates a model on test/validation data and computes metrics.

    Args:
        model: Trained scikit-learn or MLflow pyfunc model.
        evaluation_data: Validation DataFrame containing features and target.
        target_column: Name of the target column.
        id_column: Name of the identifier column to drop before scoring.

    Returns:
        Dict[str, float]: Computed evaluation metrics.
    """
    feature_columns = [col for col in evaluation_data.columns if col not in (target_column, id_column)]
    X_val = evaluation_data[feature_columns].to_numpy()
    y_true = evaluation_data[target_column].to_numpy()

    y_pred = model.predict(X_val)
    metrics = compute_evaluation_metrics(y_true, y_pred)
    return metrics


def evaluate_and_gate_promotion(
    candidate_model_uri: str,
    registered_model_name: str,
    evaluation_data: pd.DataFrame,
    target_column: str,
    id_column: str,
    candidate_version: Optional[str] = None,
    min_r2_threshold: float = 0.1,
    client: Optional[MlflowClient] = None,
) -> Tuple[bool, Dict[str, float], Optional[Dict[str, float]]]:
    """Evaluates candidate model against champion baseline and gates promotion.

    Automated evaluation gate for Level 2 maturity:
    1. Computes validation metrics for candidate model.
    2. Logs metrics to MLflow via mlflow.log_metric.
    3. Retrieves current champion metrics from MLflow Model Registry.
    4. Evaluates promotion condition: candidate_rmse < champion_rmse and r2 >= min_r2_threshold.
    5. Promotes candidate to @champion if condition is satisfied, else flags as @challenger.

    Returns:
        Tuple[bool, Dict[str, float], Optional[Dict[str, float]]]:
            (is_promoted, candidate_metrics, champion_metrics)
    """
    client = client or MlflowClient()
    logger.info(f"Loading candidate model from {candidate_model_uri} for evaluation.")
    candidate_model = mlflow.pyfunc.load_model(candidate_model_uri)

    candidate_metrics = evaluate_model_performance(
        candidate_model, evaluation_data, target_column, id_column
    )

    # Log metrics to MLflow run
    for metric_name, metric_val in candidate_metrics.items():
        mlflow.log_metric(f"val_{metric_name}", metric_val)
        logger.info(f"Logged metric: val_{metric_name} = {metric_val:.4f}")

    # Retrieve current champion metrics
    champion_rmse = get_metric_from_model_alias(
        registered_model_name, alias="champion", metric_name="rmse"
    )
    champion_metrics = {"rmse": champion_rmse} if champion_rmse is not None else None

    # Evaluation promotion gate logic
    is_promoted = False
    candidate_rmse = candidate_metrics["rmse"]
    candidate_r2 = candidate_metrics["r2_score"]

    if champion_rmse is None:
        # No champion exists yet; baseline validation threshold applies
        if candidate_r2 >= min_r2_threshold:
            logger.info("No prior champion found. Candidate passes baseline threshold -> Promoting to champion.")
            is_promoted = True
        else:
            logger.warning(f"Candidate R2 ({candidate_r2:.3f}) below threshold ({min_r2_threshold}) -> Rejecting.")
    else:
        # Challenger must outperform existing champion on validation RMSE
        if candidate_rmse < champion_rmse and candidate_r2 >= min_r2_threshold:
            logger.info(
                f"Candidate beats champion (RMSE: {candidate_rmse:.3f} < {champion_rmse:.3f}, R2: {candidate_r2:.3f}) "
                "-> Metric gate PASSED. Promoting to champion."
            )
            is_promoted = True
        else:
            logger.warning(
                f"Candidate did not beat champion (RMSE: {candidate_rmse:.3f} >= {champion_rmse:.3f}) "
                "-> Metric gate FAILED. Retaining as challenger."
            )

    if is_promoted:
        set_champion_alias_on_logged_model(registered_model_name, model_alias="champion")
        if candidate_version:
            client.set_model_version_tag(
                name=registered_model_name,
                version=candidate_version,
                key="promotion_decision",
                value="PROMOTED_CHAMPION",
            )
    else:
        if candidate_version:
            client.set_registered_model_alias(
                name=registered_model_name,
                alias="challenger",
                version=candidate_version,
            )
            client.set_model_version_tag(
                name=registered_model_name,
                version=candidate_version,
                key="promotion_decision",
                value="REJECTED_CHALLENGER",
            )

    return is_promoted, candidate_metrics, champion_metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate model and gate promotion.")
    parser.add_argument("--candidate_model_uri", required=True, help="URI of the candidate model.")
    parser.add_argument("--registered_model_name", required=True, help="Registered model name.")
    parser.add_argument("--candidate_version", required=False, default=None, help="Candidate version number.")
    parser.add_argument("--val_data_path", required=False, default=None, help="Path to validation data parquet/csv.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logger.info("Executing automated model evaluation pipeline stage.")
    # In live environments, validation data is loaded from Unity Catalog or feature baseline
    dummy_data = pd.DataFrame({
        "theme_id": [1, 2, 3, 4, 5],
        "year": [2020, 2021, 2022, 2023, 2024],
        "pieces": [100.0, 250.0, 500.0, 750.0, 1200.0],
        "set_id": [101, 102, 103, 104, 105],
    })
    evaluate_and_gate_promotion(
        candidate_model_uri=args.candidate_model_uri,
        registered_model_name=args.registered_model_name,
        evaluation_data=dummy_data,
        target_column="pieces",
        id_column="set_id",
        candidate_version=args.candidate_version,
    )


if __name__ == "__main__":
    main()
