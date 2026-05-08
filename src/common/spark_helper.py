from pyspark.sql import DataFrame, SparkSession
from delta.tables import DeltaTable
from databricks.feature_store import FeatureStoreClient

from typing import List

from common.utils import get_logger

logger = get_logger()

fs = FeatureStoreClient()

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

def register_delta_table_in_feature_store(spark: SparkSession,
                                          fully_qualified_path: str, 
                                          primary_keys: List[str]
                                          ) -> None:
    '''Registers a Delta table in the Databricks Feature Store with specified primary keys.

    Usage: Feature Store registration is needed when the table should be visible in the Feature Store tab on the Databricks Web UI, or when you want to enforce one or more primary keys on the table.

    Args:
        spark (SparkSession): The active Spark session.
        fully_qualified_path (str): The fully qualified name of the Delta table (e.g., "catalog.schema.table").
        primary_keys (List[str]): The list of primary key columns for the table.
    '''
    
    try: 
        fs.get_table(name = fully_qualified_path)
        logger.info(f"Artifact already registered in feature store ({fully_qualified_path})!")
        
    except: 
        table_name = fully_qualified_path.split(".")[-1]

        for primary_key in primary_keys: 
            
            spark.sql(f"ALTER TABLE {fully_qualified_path} ALTER COLUMN {primary_key} SET NOT NULL")

            spark.sql(f"ALTER TABLE {fully_qualified_path} ADD CONSTRAINT {table_name}_pk PRIMARY KEY({primary_key})")
        
        fs.register_table(delta_table=fully_qualified_path,
                          primary_keys=primary_keys,
                          description="Anime scoring features. This contains anime Name, genre and score.")  
        logger.info(f"Artifact registered in feature store ({fully_qualified_path})!")