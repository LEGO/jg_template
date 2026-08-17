import argparse

import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import ArrayType, DoubleType, StringType, LongType, StructField, StructType

from common.spark_helper import get_spark_session, upsert_delta_table
from common.utils import get_logger

logger = get_logger()

# Schema of the `response` JSON blob captured by AutoCaptureConfigInput.
# The served model returns {"predictions": [<double>, ...]}.
_RESPONSE_SCHEMA = StructType([StructField("predictions", ArrayType(DoubleType()), True)])

# Schema of the `_payload` source table written by Databricks model serving.
_PAYLOAD_SCHEMA = StructType([
    StructField("databricks_request_id", StringType(), True),
    StructField("timestamp_ms", LongType(), True),
    StructField("request", StringType(), True),
    StructField("response", StringType(), True),
])


def unpack_payload(df: DataFrame) -> DataFrame:
    """Flatten an inference `_payload` DataFrame into one typed row per prediction.

    Args:
        df: DataFrame with columns ``databricks_request_id`` (str),
            ``timestamp_ms`` (long), ``request`` (JSON str), ``response`` (JSON str).

    Returns:
        DataFrame with columns ``record_id``, ``prediction_ts`` (timestamp),
        ``Predicted_Score`` (double), ``model_version`` (str). One row per element
        of ``response.predictions``; ``record_id`` is ``"{request_id}-{index}"``.
    """
    parsed = df.withColumn("_resp", F.from_json(F.col("response"), _RESPONSE_SCHEMA))
    exploded = parsed.select(
        F.col("databricks_request_id").alias("_req_id"),
        (F.col("timestamp_ms") / 1000).cast("timestamp").alias("prediction_ts"),
        F.posexplode(F.col("_resp.predictions")).alias("_idx", "Predicted_Score"),
    )
    return exploded.select(
        F.concat_ws("-", F.col("_req_id"), F.col("_idx").cast("string")).alias("record_id"),
        F.col("prediction_ts"),
        F.col("Predicted_Score").cast("double").alias("Predicted_Score"),
        F.lit("unknown").alias("model_version"),
    )


def empty_unpacked_table(spark: SparkSession) -> DataFrame:
    """Return an empty DataFrame with the correct unpacked-predictions schema.

    Produces the same schema as ``unpack_payload`` by passing an empty
    ``_payload``-shaped DataFrame through the transform.  Use this to
    pre-create the target table when no source data is available yet, so
    downstream tasks (e.g. ``setup_monitor``) always have a table to attach to.

    Args:
        spark: Active Spark session.

    Returns:
        Empty DataFrame with columns ``record_id``, ``prediction_ts``,
        ``Predicted_Score``, and ``model_version``.
    """
    empty_source = spark.createDataFrame([], _PAYLOAD_SCHEMA)
    return unpack_payload(empty_source)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unpack serving inference payload table.")
    parser.add_argument("--catalog", required=True, help="Catalog name.")
    parser.add_argument("--schema", required=True, help="Schema name.")
    parser.add_argument(
        "--endpoint_name",
        required=True,
        help="Serving endpoint name; source table is <endpoint_name>_payload.",
    )
    parser.add_argument(
        "--unpacked_table",
        required=True,
        help="Name of the flat predictions table to write.",
    )
    return parser.parse_args()


def main() -> None:
    """Read <endpoint_name>_payload, flatten it, and upsert the unpacked table."""
    args = _parse_args()
    spark = get_spark_session("unpack_inference_table")

    source = f"{args.catalog}.{args.schema}.{args.endpoint_name}_payload"
    target = f"{args.catalog}.{args.schema}.{args.unpacked_table}"

    if not spark.catalog.tableExists(source):
        logger.info(f"Source table {source} does not exist yet (no traffic). Nothing to unpack.")
        logger.info(f"Creating empty target table {target} so downstream monitor setup has a target.")
        unpacked = empty_unpacked_table(spark)
        upsert_delta_table(spark, unpacked, target, primary_key="record_id")
        return

    df = spark.read.table(source)
    if df.limit(1).count() == 0:
        logger.info(f"Source table {source} is empty. Nothing to unpack.")
        logger.info(f"Creating empty target table {target} so downstream monitor setup has a target.")
        unpacked = empty_unpacked_table(spark)
        upsert_delta_table(spark, unpacked, target, primary_key="record_id")
        return

    unpacked = unpack_payload(df)
    logger.info(f"Writing unpacked predictions to {target}")
    upsert_delta_table(spark, unpacked, target, primary_key="record_id")


if __name__ == "__main__":
    main()
