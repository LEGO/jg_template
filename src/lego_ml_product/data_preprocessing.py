import re
from pathlib import Path

import mlflow
from mlflow.data.spark_dataset import from_spark
from pyspark.sql import DataFrame
from pyspark.sql.types import DoubleType, FloatType, IntegerType
import pyspark.sql.functions as F

import argparse
from typing import Dict, List, Tuple
from omegaconf import OmegaConf
from common.mlflow_helper import start_mlflow_experiment_and_run
from common.spark_helper import get_spark_session, upsert_delta_table, register_delta_table_in_feature_store
from common.utils import get_logger, load_model_config

logger = get_logger()

'''
The data preprocessing script is compliant with level 2 of https://baseplate.legogroup.io/catalog/default/component/ds_ai_handbook/docs/traditional_ml/docs/maturity_levels/1-mlops-data-preparation/#data-preparation 
The data preprocessing script is compliant with level 2 of https://baseplate.legogroup.io/catalog/default/component/ds_ai_handbook/docs/traditional_ml/docs/maturity_levels/8-mlops-feature-store/#feature-store.
The data preprocessing script is (arguably) compliant with level 2 of https://baseplate.legogroup.io/catalog/default/component/ds_ai_handbook/docs/traditional_ml/docs/maturity_levels/10-mlops-data-monitoring/.

Explanation: 
Data Preprocessing 1) The data preparation steps are modularized and decoupled from model training, and easily tested, i.e. fix_data_types and create_category_onehot_encodings functions. This allows for parallel execution and reusability of the data preparation steps across different pipelines.
Data Preprocessing 2) Features are stored in a unity catalog that serves as a "feature store", making them readily available for training and inference.
Feature Store 1) The unity catalog table is updated on schedule as the model demands, ensuring that the most up-to-date features are available for training and inference.
Data Monitoring 1) Basic data monitoring is implemented by logging the number of rows preprocessed to MLflow. This allows for tracking changes in the data volume over time, which can be an indicator of data quality issues or changes in the underlying data distribution. However, a production implementation should include much more comprehensive data monitoring, e.g. including monitoring of feature drift, data quality etc.
'''


def fix_data_types(dataframe: DataFrame, target_name: str, numeric_features: List[str]) -> DataFrame:
    """Casts the string-typed bronze columns to numerics.

    Integers go via double: under ANSI mode a direct string -> int cast on "1965.0"
    fails. Expects drop_invalid_rows to have run first.
    """
    dataframe = dataframe.withColumn(target_name, F.col(target_name).cast(FloatType()))
    for column in numeric_features:
        dataframe = dataframe.withColumn(column, F.col(column).cast(DoubleType()).cast(IntegerType()))
    return dataframe


def drop_invalid_rows(dataframe: DataFrame, id_column: str, target_name: str) -> DataFrame:
    """Drops rows with no usable id or target, while both are still raw text.

    Empty bronze fields are empty strings, not NULL, so both are checked.
    """
    def _present(column: str):
        return F.col(column).isNotNull() & (F.trim(F.col(column)) != "")

    return dataframe.where(_present(id_column) & _present(target_name))


def filter_buildable_sets(dataframe: DataFrame, target_name: str) -> DataFrame:
    """Keeps only sets that contain parts, dropping zero-part merchandise."""
    return dataframe.where(F.col(target_name) > 0)


def deduplicate_on_key(dataframe: DataFrame, id_column: str) -> DataFrame:
    """Leaves one row per id.

    ``upsert_delta_table`` MERGEs on this key, and Delta MERGE fails on duplicates.
    """
    return dataframe.dropDuplicates([id_column])


def to_column_name(value: str) -> str:
    """Turns a category value into a Spark-safe column name.

    Any run of non-alphanumerics collapses to one underscore. A '.' in a column name is
    read by Spark as nested-field access, so "4.5V" would break ``select``.
    """
    return re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9]+", "_", value.strip())).strip("_")


def build_category_vocabulary(
    dataframe: DataFrame,
    categorical: str,
    top_n: int,
    other_label: str = "Other",
) -> List[str]:
    """Returns the ``top_n`` most common category values, plus the residual label.

    Call on the WHOLE dataset, before any year window, or each windowed run gets a
    different set of columns. Ties break on value ascending so the result is reproducible.
    ``other_label`` is appended only if it did not already win a slot on count.
    """
    rows = (
        dataframe.groupBy(categorical)
        .count()
        .orderBy(F.col("count").desc(), F.col(categorical).asc())
        .limit(top_n)
        .collect()
    )
    vocabulary = [row[categorical] for row in rows]
    if other_label not in vocabulary:
        vocabulary.append(other_label)
    return vocabulary


def create_category_onehot_encodings(
    dataframe: DataFrame,
    categorical: str,
    vocabulary: List[str],
    other_label: str = "Other",
) -> Tuple[DataFrame, List[str]]:
    """One-hot encodes ``categorical`` against a fixed vocabulary.

    Values outside the vocabulary map to ``other_label``, so a category never seen at
    training time degrades gracefully instead of failing at inference.

    Raises:
        ValueError: If two different category values map to the same column name.
    """
    column_names: Dict[str, str] = {}
    for value in vocabulary:
        name = to_column_name(value)
        if name in column_names:
            raise ValueError(
                f"Category column collision: {value!r} and {column_names[name]!r} "
                f"both map to column {name!r}."
            )
        column_names[name] = value

    dataframe = dataframe.withColumn(
        "_bucketed_category",
        F.when(F.col(categorical).isin(vocabulary), F.col(categorical)).otherwise(F.lit(other_label)),
    )

    encoded_columns: List[str] = []
    for name, value in column_names.items():
        dataframe = dataframe.withColumn(
            name, F.when(F.col("_bucketed_category") == value, 1).otherwise(0)
        )
        encoded_columns.append(name)

    return dataframe.drop("_bucketed_category"), encoded_columns


def apply_year_window(
    dataframe: DataFrame,
    year_column: str,
    min_year: int | None,
    max_year: int | None,
) -> DataFrame:
    """Restricts rows to a release-year window.

    Leave both bounds unset for the full dataset. Apply AFTER the vocabulary is built.
    """
    if min_year is not None:
        dataframe = dataframe.where(F.col(year_column) >= min_year)
    if max_year is not None:
        dataframe = dataframe.where(F.col(year_column) <= max_year)
    return dataframe


def build_feature_frame(
    dataframe: DataFrame,
    column_params: dict,
    min_year: int | None,
    max_year: int | None,
) -> Tuple[DataFrame, List[str]]:
    """Runs the ordered feature pipeline and returns the frame plus its one-hot columns.

    Two orderings are load-bearing; reversing either fails silently:
    drop invalid rows BEFORE casting, and build the vocabulary BEFORE the year window.
    """
    id_column = column_params["id"]
    target_name = column_params["target_name"]
    categorical = column_params["categorical"]
    numeric_features = column_params["numeric_features"]

    dataframe = drop_invalid_rows(dataframe, id_column=id_column, target_name=target_name)
    dataframe = fix_data_types(dataframe, target_name=target_name, numeric_features=numeric_features)
    dataframe = filter_buildable_sets(dataframe, target_name=target_name)
    dataframe = deduplicate_on_key(dataframe, id_column=id_column)

    vocabulary = build_category_vocabulary(
        dataframe, categorical=categorical, top_n=column_params["top_n_categories"]
    )
    logger.info("Category vocabulary size: %s", len(vocabulary))

    dataframe = apply_year_window(dataframe, column_params["year_column"], min_year, max_year)
    dataframe, encoded_columns = create_category_onehot_encodings(
        dataframe, categorical=categorical, vocabulary=vocabulary
    )

    dataframe = dataframe.select(id_column, target_name, *numeric_features, *encoded_columns)
    return dataframe, encoded_columns


def write_baseline_if_absent(spark, dataframe: DataFrame, baseline_table_name: str) -> bool:
    """Writes the monitor's reference snapshot once, and never touches it again.

    Re-baselining means dropping the table.

    Returns:
        True if the baseline was created by this call, False if it already existed.
    """
    if spark.catalog.tableExists(baseline_table_name):
        logger.info("Baseline table %s already exists; leaving it untouched.", baseline_table_name)
        return False

    dataframe.write.format("delta").mode("overwrite").saveAsTable(baseline_table_name)
    logger.info("Created baseline table %s.", baseline_table_name)
    return True


def _sql_comment_literal(comment: str) -> str:
    """Renders comment text as a single-quoted SQL literal, escaping embedded quotes.

    Spark escapes a single quote by doubling it; an unescaped one raises
    PARSE_SYNTAX_ERROR. Category values can contain apostrophes.
    """
    return "'" + comment.replace("'", "''") + "'"


def alter_table_with_comments(
    spark,
    fully_qualified_feature_table_name: str,
    column_params: dict,
    encoded_columns: List[str],
) -> None:
    """Applies column comments to the feature table."""
    logger.info("Applying column comments to feature table.")
    table = fully_qualified_feature_table_name
    categorical = column_params["categorical"]
    other_label = "Other"

    def _comment(column: str, comment: str) -> None:
        """Emits one ALTER COLUMN, backticking the identifier and escaping the literal."""
        spark.sql(
            f"ALTER TABLE {table} ALTER COLUMN `{column}` "
            f"COMMENT {_sql_comment_literal(comment)}"
        )

    _comment(column_params["id"], "LEGO set number, used as the primary key.")
    _comment(column_params["target_name"], "Number of parts in the set. Prediction target.")
    for column in column_params.get("numeric_features", []):
        _comment(column, f"Numeric feature: {column}.")
    for column in encoded_columns:
        if column == other_label:
            comment = (
                f"One-hot encoded {categorical}: the source's own '{other_label}' theme "
                f"combined with every theme outside the top "
                f"{column_params['top_n_categories']}."
            )
        else:
            comment = f"One-hot encoded value for {categorical}: {column}."
        _comment(column, comment)


def _optional_year(value: str | None) -> int | None:
    """Normalises a year CLI argument to an int, or None when unset.

    Bundle variables default to empty, so the task gets `--min_year=`. `type=int` on the
    argument would make argparse exit non-zero on that.
    """
    if value is None or str(value).strip() == "":
        return None
    return int(value)


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
        "Decoupled from the write catalog/schema: the source may be another team's "
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
    parser.add_argument(
        "--min_year",
        required=False,
        default=None,
        help="Earliest release year to include. Drift-replay lever; empty or omitted means all years.",
    )
    parser.add_argument(
        "--max_year",
        required=False,
        default=None,
        help="Latest release year to include. Drift-replay lever; empty or omitted means all years.",
    )
    parser.add_argument(
        "--baseline_table_name",
        required=False,
        default=None,
        help="Name of the monitor's baseline table. Written once, on the first run.",
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

    config_path = Path(__file__).parent / "model" / "model_config.yml"
    column_params: dict = OmegaConf.to_container(load_model_config(config_path).columns, resolve=True)
    id_column = column_params["id"]

    fully_qualified_data_table_path = args.source_table
    logger.info(f"Reading data from {fully_qualified_data_table_path}")
    dataframe = spark.table(fully_qualified_data_table_path)
    mlflow.log_param("data_source_table", fully_qualified_data_table_path)

    min_year = _optional_year(args.min_year)
    max_year = _optional_year(args.max_year)

    dataframe, encoded_columns = build_feature_frame(dataframe, column_params, min_year, max_year)

    mlflow.log_param(
        "preprocessing_steps",
        [
            drop_invalid_rows.__name__,
            fix_data_types.__name__,
            filter_buildable_sets.__name__,
            deduplicate_on_key.__name__,
            apply_year_window.__name__,
            create_category_onehot_encodings.__name__,
        ],
    )
    mlflow.log_params({"min_year": min_year, "max_year": max_year})

    logger.info("Data preprocessing complete. Writing preprocessed data to feature store.")

    mlflow.log_input(from_spark(dataframe), context="preprocessed_data")
    mlflow.log_metric("num_rows_preprocessed", dataframe.count()) #NOTE this is a naive implementation of data monitoring and should be much more comprehensive in a production scenario, e.g. including monitoring of feature drift, data quality etc.

    fully_qualified_feature_table_name = (
        f"{args.catalog_name}.{args.schema_name}.{args.feature_store_table_name}"
    )
    logger.info(f"Upserting feature table: {fully_qualified_feature_table_name}")
    upsert_delta_table(spark, dataframe, fully_qualified_feature_table_name, primary_key=id_column)
    mlflow.log_param("feature_table", fully_qualified_feature_table_name)

    # Feature Store Level 2: Register feature table in Databricks Feature Store
    logger.info(f"Registering table in Databricks Feature Store: {fully_qualified_feature_table_name}")
    register_delta_table_in_feature_store(
        spark=spark,
        fully_qualified_path=fully_qualified_feature_table_name,
        primary_key=id_column,
    )

    if args.baseline_table_name:
        baseline_fqn = f"{args.catalog_name}.{args.schema_name}.{args.baseline_table_name}"
        created = write_baseline_if_absent(spark, dataframe, baseline_fqn)
        mlflow.log_param("baseline_table", baseline_fqn)
        mlflow.log_param("baseline_created_this_run", created)

    alter_table_with_comments(
        spark, fully_qualified_feature_table_name, column_params, encoded_columns
    )

    mlflow.end_run()

if __name__ == "__main__":
    main()
