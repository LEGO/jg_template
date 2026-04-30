from pyspark.sql import DataFrame
from pathlib import Path
import pyspark.sql.functions as F

import mlflow 

import argparse

from common.mlflow_helper import start_mlflow_experiment_and_run
from common.spark_helper import get_spark_session, upsert_delta_table
from common.utils import get_logger

logger = get_logger()


def pick_random_subset(dataframe: DataFrame, seed: int = 42) -> DataFrame:
    """Picks a random subset of the preprocessed data for batch prediction."""
    return dataframe.orderBy(F.rand(seed=seed)).limit(10)

def _parse_args() -> argparse.Namespace:
    """Parses and returns CLI arguments for the ingestion pipeline.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run the model training pipeline.",
    )
    parser.add_argument(
        "--catalog_name",
        required=True,
        help="Catalog name where the training table will be written.",
    )
    parser.add_argument(
        "--schema_name",
        required=True,
        help="Schema (database) name where the training table will be written.",
    )
    parser.add_argument(
        "--feature_store_table_name",
        required=True,
        help="Name of the feature table to be registered.",
    )
    parser.add_argument(
        "--batch_prediction_table",
        required=True,
        help="Name of the table where batch predictions will be written.",
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
    return parser.parse_args()

def main() -> None:
    """Entry point for the batch prediction script.

    Parses arguments, initializes Spark, and runs the batch prediction pipeline.
    """
    logger.info("Starting batch prediction pipeline.")
    args = _parse_args()
    
    spark = get_spark_session(Path(__file__).stem )

    fully_qualified_feature_table_name = f"{args.catalog_name}.{args.schema_name}.{args.feature_store_table_name}"
    logger.info(f"Reading preprocessed data from feature store table: {fully_qualified_feature_table_name}")
    dataframe = spark.read.table(fully_qualified_feature_table_name)

    dataframe = pick_random_subset(dataframe)
    
    start_mlflow_experiment_and_run(experiment_path=args.experiment_path)
    
    logger.info(f"Loading model {args.catalog_name}.{args.schema_name}.{args.model_name}@champion from MLflow Model Registry")
    champion_model = mlflow.pyfunc.load_model(model_uri=f"models:/{args.catalog_name}.{args.schema_name}.{args.model_name}@champion")

    #NOTE converting to Pandas for sklearn to work
    pandas_df = dataframe.toPandas()
    pandas_df["Predicted_Score"] = champion_model.predict(pandas_df.drop(["Name", "Score"], axis=1).to_numpy())
    dataframe = spark.createDataFrame(pandas_df)
    
    logger.info(f"Writing batch predictions to {args.catalog_name}.{args.schema_name}.{args.batch_prediction_table}")
    upsert_delta_table(spark, dataframe, f"{args.catalog_name}.{args.schema_name}.{args.batch_prediction_table}", primary_key="Name")
if __name__ == "__main__":
    main()