from pathlib import Path

import mlflow
from mlflow.data.spark_dataset import from_spark
from pyspark.sql import DataFrame
from pyspark.sql.types import FloatType
import pyspark.sql.functions as F

import argparse
from typing import List, Tuple
from common.mlflow_helper import start_mlflow_experiment_and_run
from common.spark_helper import get_spark_session, upsert_delta_table
from common.utils import get_logger

logger = get_logger()


def fix_data_types(dataframe: DataFrame) -> DataFrame:
    """Fixes data types in the source table to ensure compatibility with the feature model training pipeline."""
    dataframe = dataframe.withColumn("Score", F.col("Score").cast(FloatType()))
    return dataframe

def create_genre_onehot_encodings(dataframe: DataFrame) -> Tuple[DataFrame, List[str]]:
    """Creates genre onehot encodings from the Genre column in the source table."""
    # Collect all unique genres
    all_genres = (
        dataframe.select(F.explode(F.split(F.col("Genres"), ",")).alias("genre"))
        .select(F.trim(F.col("genre")).alias("genre"))
        .distinct()
        .rdd.flatMap(lambda x: x)
        .collect()
    )

    # One-hot encode each genre as a binary column
    clean_genres = list()
    for genre in all_genres:
        dataframe = dataframe.withColumn(
            genre.strip().replace(" ", "_"),
            F.when(F.array_contains(F.split(F.col("Genres"), r",\s*"), genre), 1).otherwise(0)
        )
        clean_genres.append(genre.strip().replace(" ", "_"))

    return dataframe, clean_genres

def _parse_args() -> argparse.Namespace:
    """Parses and returns CLI arguments for the ingestion pipeline.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run the data preprocessing pipeline.",
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
        "--table_name",
        required=True,
        help="Name of the Delta table containing the training data.",
    )
    parser.add_argument(
        "--feature_store_table_name",
        required=True,
        help="Name of the feature table to be registered.",
    )
    parser.add_argument(
        "--experiment_path",
        required=True,
        help="Path of the experiment to be used for logging.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the data preprocessing script.

    Parses arguments, initializes Spark, and runs the data preprocessing pipeline.
    """
    
    logger.info("Starting data preprocessing pipeline.")
    args = _parse_args()

    logger.info("Logging to mlflow experiment at: %s", args.experiment_path)
    start_mlflow_experiment_and_run(experiment_path=args.experiment_path)
    
    logger.info("Initializing Spark session.")
    spark = get_spark_session(Path(__file__).stem)

    fully_qualified_data_table_path = f"{args.catalog_name}.{args.schema_name}.{args.table_name}"
    logger.info(f"Reading data from {fully_qualified_data_table_path}")
    dataframe = spark.table(fully_qualified_data_table_path)
    mlflow.log_param("data_source_table", fully_qualified_data_table_path)

    dataframe = fix_data_types(dataframe=dataframe)
    dataframe, genres = create_genre_onehot_encodings(dataframe=dataframe)
    mlflow.log_param("preprocessing_steps", [fix_data_types.__name__, 
                                             create_genre_onehot_encodings.__name__])

    dataframe = dataframe.select("Name", "Score", *genres)
    logger.info("Data preprocessing complete. Writing preprocessed data to feature store.")
    
    mlflow.log_input(from_spark(dataframe), context="preprocessed_data")

    fully_qualified_feature_table_name = f"{args.catalog_name}.{args.schema_name}.{args.feature_store_table_name}"
    logger.info(f"Upserting feature table: {fully_qualified_feature_table_name}")
    upsert_delta_table(spark, dataframe, fully_qualified_feature_table_name, primary_key="Name")
    mlflow.log_param("feature_table", fully_qualified_feature_table_name)

    mlflow.end_run()

if __name__ == "__main__":
    main()
