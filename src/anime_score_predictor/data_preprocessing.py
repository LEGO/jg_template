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

'''
The data preprocessing script is compliant with level 2 of https://baseplate.legogroup.io/catalog/default/component/ds_ai_handbook/docs/traditional_ml/docs/maturity_levels/1-mlops-data-preparation/#data-preparation 
The data preprocessing script is compliant with level 2 of https://baseplate.legogroup.io/catalog/default/component/ds_ai_handbook/docs/traditional_ml/docs/maturity_levels/8-mlops-feature-store/#feature-store.
The data preprocessing script is (arguably) compliant with level 2 of https://baseplate.legogroup.io/catalog/default/component/ds_ai_handbook/docs/traditional_ml/docs/maturity_levels/10-mlops-data-monitoring/.

Explanation: 
Data Preprocessing 1) The data preparation steps are modularized and decoupled from model training, and easily tested, i.e. fix_data_types and create_genre_onehot_encodings functions. This allows for parallel execution and reusability of the data preparation steps across different pipelines.
Data Preprocessing 2) Features are stored in a unity catalog that serves as a "feature store", making them readily available for training and inference.
Feature Store 1) The unity catalog table is updated on schedule as the model demands, ensuring that the most up-to-date features are available for training and inference.
Data Monitoring 1) Basic data monitoring is implemented by logging the number of rows preprocessed to MLflow. This allows for tracking changes in the data volume over time, which can be an indicator of data quality issues or changes in the underlying data distribution. However, a production implementation should include much more comprehensive data monitoring, e.g. including monitoring of feature drift, data quality etc.
'''


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


def alter_table_with_comments(spark, fully_qualified_feature_table_name, genres) -> None:
    
    logger.info("Applying column comments to feature table.")
    spark.sql(
        f"ALTER TABLE {fully_qualified_feature_table_name} ALTER COLUMN Name COMMENT 'Anime name, used as the primary key.'"
    )
    spark.sql(f"ALTER TABLE {fully_qualified_feature_table_name} ALTER COLUMN Score COMMENT 'Anime review score.'")
    for genre in genres:
        spark.sql(
            f"ALTER TABLE {fully_qualified_feature_table_name} ALTER COLUMN `{genre}` COMMENT 'One-hot encoded value for genre: {genre}.'"
        )

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
        help="Schema (database) name where the feature table will be written.",
    )
    parser.add_argument(
        "--source_table",
        required=True,
        help="Fully-qualified source table to read, e.g. 'catalog.schema.table'. "
        "Decoupled from the write schema: the source may be another team's "
        "schema, another catalog, etc. — not necessarily where features are written.",
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

    fully_qualified_data_table_path = args.source_table
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
    mlflow.log_metric("num_rows_preprocessed", dataframe.count()) #NOTE this is a naive implementation of data monitoring and should be much more comprehensive in a production scenario, e.g. including monitoring of feature drift, data quality etc.

    fully_qualified_feature_table_name = f"{args.catalog_name}.{args.schema_name}.{args.feature_store_table_name}"
    logger.info(f"Upserting feature table: {fully_qualified_feature_table_name}")
    upsert_delta_table(spark, dataframe, fully_qualified_feature_table_name, primary_key="Name")
    mlflow.log_param("feature_table", fully_qualified_feature_table_name)

    ''' 
    Registering the feature table in the Databricks Feature Store makes the table visible in the Feature Store tab on the Web UI. The underlying table is still a Delta table in the Unity Catalog. 
    Adding it to the code is optional and can be done with common.spark_helper.register_delta_table_in_feature_store. 
    '''
     
    alter_table_with_comments(spark, fully_qualified_feature_table_name, genres)

    mlflow.end_run()

if __name__ == "__main__":
    main()
