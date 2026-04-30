from pyspark.sql import DataFrame, SparkSession
from delta.tables import DeltaTable


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