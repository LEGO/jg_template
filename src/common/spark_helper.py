from pyspark.sql import DataFrame, SparkSession
from delta.tables import DeltaTable
from databricks.feature_engineering import FeatureEngineeringClient

from typing import List

from common.utils import get_logger

logger = get_logger()

def get_spark_session(name: str) -> SparkSession:
    """Gets the active Spark session or creates a new one if none exists.

    Args:
        name (str): The name of the Spark application.

    Returns:
        SparkSession: The active or newly created Spark session.
    """
    spark = SparkSession.getActiveSession()
    if spark is None:
        spark = SparkSession.builder.appName(name).getOrCreate()

    return spark 

def upsert_delta_table(spark: SparkSession, dataframe: DataFrame, fully_qualified_table_name: str, primary_key: str):
    """Writes a DataFrame to a Delta table, overwriting if it exists or creating it if it doesn't.

    Args:
        dataframe (DataFrame): The DataFrame to write.
        fully_qualified_table_name (str): The fully qualified name of the Delta table (e.g., "catalog.schema.table").
        primary_key (str): The primary key column used for upsert operations.
    """
    target_exists = spark.catalog.tableExists(fully_qualified_table_name)

    if not target_exists:
        dataframe.write.format("delta").mode("overwrite").saveAsTable(fully_qualified_table_name)
    
    elif primary_key not in spark.read.table(fully_qualified_table_name).columns:
        dataframe.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(fully_qualified_table_name)
    
    else:
        delta_table = DeltaTable.forName(spark, fully_qualified_table_name)
        
        delta_table.alias("target").merge(
            dataframe.alias("source"),
            f"target.{primary_key} = source.{primary_key}"
        ).whenMatchedUpdateAll() \
         .whenNotMatchedInsertAll() \
         .execute()

def register_delta_table_in_feature_store(
    spark: SparkSession,
    fully_qualified_path: str,
    primary_keys: List[str],
    description: str | None = None,
    tags: dict[str, str] | None = None,
) -> None:
    """Register a Unity Catalog Delta table as a feature table.

    In Unity Catalog, a Delta table with PRIMARY KEY constraints in the right
    location is already a feature table. ``fe.create_table`` attaches the
    description/tags metadata and exposes the table in the Databricks Features UI.

    Args:
        spark: Active Spark session.
        fully_qualified_path: Fully qualified UC name, e.g. ``catalog.schema.table``.
            The table must already exist as a UC Delta table (typically written by
            ``upsert_delta_table``).
        primary_keys: Columns to mark ``NOT NULL`` and constrain as the primary key.
        description: Optional human-readable description shown in the Features UI.
        tags: Optional governance/lineage tags attached to the feature table.
    """
    fe = FeatureEngineeringClient()

    try:
        fe.get_table(name=fully_qualified_path)
        logger.info(f"Artifact already registered in feature store ({fully_qualified_path})!")
        return
    except ValueError:
        pass

    table_name = fully_qualified_path.split(".")[-1]
    for primary_key in primary_keys:
        spark.sql(f"ALTER TABLE {fully_qualified_path} ALTER COLUMN {primary_key} SET NOT NULL")
        spark.sql(f"ALTER TABLE {fully_qualified_path} ADD CONSTRAINT {table_name}_pk PRIMARY KEY({primary_key})")

    fe.create_table(
        name=fully_qualified_path,
        primary_keys=primary_keys,
        source=fully_qualified_path,
        description=description,
        tags=tags,
    )
    logger.info(f"Artifact registered in feature store ({fully_qualified_path})!")