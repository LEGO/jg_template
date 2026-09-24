import argparse

import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from common.spark_helper import get_spark_session, upsert_delta_table
from common.utils import get_logger

logger = get_logger()

# Schema of the `response` JSON blob captured in the AI Gateway inference table.
# The served model returns {"predictions": [<double>, ...]}.
_RESPONSE_SCHEMA = StructType([StructField("predictions", ArrayType(DoubleType()), True)])

# Schema of the `request` JSON blob. The tensor-signature model is called with
# {"inputs": [[<double>, ...], ...]} — one inner array of features per scored record,
# positionally aligned with `response.predictions`.
_REQUEST_SCHEMA = StructType(
    [StructField("inputs", ArrayType(ArrayType(DoubleType())), True)]
)

# Subset of the AI Gateway `_payload` inference table columns this transform reads.
# (The live table has more columns — requester, execution_duration_ms, etc. —
# but only these drive the unpacked predictions table.)
_PAYLOAD_SCHEMA = StructType([
    StructField("databricks_request_id", StringType(), True),
    StructField("request_time", TimestampType(), True),
    StructField("status_code", IntegerType(), True),
    StructField("request", StringType(), True),
    StructField("response", StringType(), True),
    StructField("served_entity_id", StringType(), True),
])


def unpack_payload(df: DataFrame) -> DataFrame:
    """Flatten an AI Gateway inference `_payload` DataFrame into one row per prediction.

    Only successful (``status_code == 200``) requests are unpacked; error rows carry
    an error blob in ``response`` rather than predictions and are dropped. The
    ``request.inputs`` feature vectors are zipped positionally with
    ``response.predictions`` so each output row keeps the exact input that produced
    its score.

    Args:
        df: DataFrame with columns ``databricks_request_id`` (str),
            ``request_time`` (timestamp), ``status_code`` (int),
            ``request`` (JSON str), ``response`` (JSON str), ``served_entity_id`` (str).

    Returns:
        DataFrame with columns ``record_id``, ``prediction_ts`` (timestamp),
        ``Predicted_number_of_parts`` (double), ``features`` (array<double> — the input
        feature vector for this record), ``model_version`` (str). One row per
        element of ``response.predictions``; ``record_id`` is
        ``"{request_id}-{index}"``. ``model_version`` is the ``served_entity_id``
        (stable per served model version), which the monitor slices drift by.
        The endpoint's request is a bare positional array, so ``features`` carries
        no per-theme names.
    """
    parsed = (
        df.filter(F.col("status_code") == 200)
        .withColumn("_resp", F.from_json(F.col("response"), _RESPONSE_SCHEMA))
        .withColumn("_req", F.from_json(F.col("request"), _REQUEST_SCHEMA))
    )
    zipped = parsed.select(
        F.col("databricks_request_id").alias("_req_id"),
        F.col("request_time").cast("timestamp").alias("prediction_ts"),
        F.col("served_entity_id").alias("_model_version"),
        F.posexplode(
            F.arrays_zip(
                F.col("_resp.predictions").alias("prediction"),
                F.col("_req.inputs").alias("features"),
            )
        ).alias("_idx", "_pair"),
    )
    return zipped.select(
        F.concat_ws("-", F.col("_req_id"), F.col("_idx").cast("string")).alias("record_id"),
        F.col("prediction_ts"),
        F.col("_pair.prediction").cast("double").alias("Predicted_number_of_parts"),
        F.col("_pair.features").alias("features"),
        F.col("_model_version").alias("model_version"),
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
        ``Predicted_number_of_parts``, and ``model_version``.
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
