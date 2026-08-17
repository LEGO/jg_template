import json

import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="module")
def spark():
    session = (
        SparkSession.builder.appName("test_unpack")
        .master("local[1]")
        .config("spark.sql.shuffle.partitions", "1")
        .getOrCreate()
    )
    yield session
    session.stop()


def _payload_row(request_id, ts_ms, features_list, predictions):
    return (
        request_id,
        ts_ms,
        json.dumps({"dataframe_records": features_list}),
        json.dumps({"predictions": predictions}),
    )


def test_unpack_explodes_one_row_per_prediction(spark):
    from anime_score_predictor.monitoring.unpack_inference_table import unpack_payload

    df = spark.createDataFrame(
        [
            _payload_row("req-1", 1_600_000_000_000, [{"Seinen": 1}, {"Seinen": 0}], [7.5, 6.0]),
            _payload_row("req-2", 1_600_000_100_000, [{"Seinen": 1}], [8.1]),
        ],
        ["databricks_request_id", "timestamp_ms", "request", "response"],
    )

    out = unpack_payload(df)
    rows = {r["record_id"]: r for r in out.collect()}

    assert set(rows) == {"req-1-0", "req-1-1", "req-2-0"}
    assert rows["req-1-0"]["Predicted_Score"] == pytest.approx(7.5)
    assert rows["req-1-1"]["Predicted_Score"] == pytest.approx(6.0)
    assert rows["req-2-0"]["Predicted_Score"] == pytest.approx(8.1)


def test_unpack_output_schema(spark):
    from anime_score_predictor.monitoring.unpack_inference_table import unpack_payload

    df = spark.createDataFrame(
        [_payload_row("req-1", 1_600_000_000_000, [{"Seinen": 1}], [7.5])],
        ["databricks_request_id", "timestamp_ms", "request", "response"],
    )

    out = unpack_payload(df)
    assert set(out.columns) == {"record_id", "prediction_ts", "Predicted_Score", "model_version"}
    dtypes = dict(out.dtypes)
    assert dtypes["prediction_ts"] == "timestamp"
    assert dtypes["Predicted_Score"] == "double"


def test_unpack_empty_input_returns_empty_with_schema(spark):
    from anime_score_predictor.monitoring.unpack_inference_table import unpack_payload

    df = spark.createDataFrame(
        [],
        "databricks_request_id string, timestamp_ms long, request string, response string",
    )

    out = unpack_payload(df)
    assert out.count() == 0
    assert set(out.columns) == {"record_id", "prediction_ts", "Predicted_Score", "model_version"}
