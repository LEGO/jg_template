import datetime
import json

import pytest
from pyspark.sql import SparkSession

_PAYLOAD_COLUMNS = [
    "databricks_request_id",
    "request_time",
    "status_code",
    "request",
    "response",
    "served_entity_id",
]


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


def _payload_row(
    request_id,
    request_time,
    predictions,
    inputs=None,
    status_code=200,
    served_entity_id="model-1",
):
    # Default: one feature vector per prediction, positionally aligned.
    if inputs is None:
        inputs = [[float(i)] for i in range(len(predictions))]
    return (
        request_id,
        request_time,
        status_code,
        json.dumps({"inputs": inputs}),
        json.dumps({"predictions": predictions}),
        served_entity_id,
    )


def test_unpack_explodes_one_row_per_prediction(spark):
    from anime_score_predictor.monitoring.unpack_inference_table import unpack_payload

    ts = datetime.datetime(2020, 9, 13, 12, 26, 40)
    df = spark.createDataFrame(
        [
            _payload_row("req-1", ts, [7.5, 6.0]),
            _payload_row("req-2", ts, [8.1]),
        ],
        _PAYLOAD_COLUMNS,
    )

    out = unpack_payload(df)
    rows = {r["record_id"]: r for r in out.collect()}

    assert set(rows) == {"req-1-0", "req-1-1", "req-2-0"}
    assert rows["req-1-0"]["Predicted_Score"] == pytest.approx(7.5)
    assert rows["req-1-1"]["Predicted_Score"] == pytest.approx(6.0)
    assert rows["req-2-0"]["Predicted_Score"] == pytest.approx(8.1)


def test_unpack_zips_features_with_predictions_by_position(spark):
    from anime_score_predictor.monitoring.unpack_inference_table import unpack_payload

    ts = datetime.datetime(2020, 9, 13, 12, 26, 40)
    # Two records in one request: each prediction keeps its own feature vector.
    df = spark.createDataFrame(
        [
            _payload_row(
                "req-1",
                ts,
                [7.5, 6.0],
                inputs=[[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
            ),
        ],
        _PAYLOAD_COLUMNS,
    )

    out = unpack_payload(df)
    rows = {r["record_id"]: r for r in out.collect()}

    assert rows["req-1-0"]["features"] == [1.0, 0.0, 1.0]
    assert rows["req-1-1"]["features"] == [0.0, 1.0, 0.0]


def test_unpack_carries_served_entity_id_as_model_version(spark):
    from anime_score_predictor.monitoring.unpack_inference_table import unpack_payload

    ts = datetime.datetime(2020, 9, 13, 12, 26, 40)
    df = spark.createDataFrame(
        [_payload_row("req-1", ts, [7.5], served_entity_id="entity-abc123")],
        _PAYLOAD_COLUMNS,
    )

    out = unpack_payload(df)
    assert out.collect()[0]["model_version"] == "entity-abc123"


def test_unpack_drops_error_rows(spark):
    from anime_score_predictor.monitoring.unpack_inference_table import unpack_payload

    ts = datetime.datetime(2020, 9, 13, 12, 26, 40)
    ok = _payload_row("req-ok", ts, [7.5])
    # A 400 error row carries an error blob in `response`, not predictions.
    err = (
        "req-err",
        ts,
        400,
        json.dumps({"inputs": [[1.0, 0.0]]}),
        json.dumps({"error_code": "BAD_REQUEST", "message": "bad shape"}),
        "model-1",
    )
    df = spark.createDataFrame([ok, err], _PAYLOAD_COLUMNS)

    out = unpack_payload(df)
    rows = {r["record_id"] for r in out.collect()}
    assert rows == {"req-ok-0"}


def test_unpack_output_schema(spark):
    from anime_score_predictor.monitoring.unpack_inference_table import unpack_payload

    ts = datetime.datetime(2020, 9, 13, 12, 26, 40)
    df = spark.createDataFrame(
        [_payload_row("req-1", ts, [7.5])],
        _PAYLOAD_COLUMNS,
    )

    out = unpack_payload(df)
    assert set(out.columns) == {
        "record_id",
        "prediction_ts",
        "Predicted_Score",
        "features",
        "model_version",
    }
    dtypes = dict(out.dtypes)
    assert dtypes["prediction_ts"] == "timestamp"
    assert dtypes["Predicted_Score"] == "double"
    assert dtypes["features"] == "array<double>"


def test_unpack_empty_input_returns_empty_with_schema(spark):
    from anime_score_predictor.monitoring.unpack_inference_table import unpack_payload

    df = spark.createDataFrame(
        [],
        "databricks_request_id string, request_time timestamp, status_code int, "
        "request string, response string, served_entity_id string",
    )

    out = unpack_payload(df)
    assert out.count() == 0
    assert set(out.columns) == {
        "record_id",
        "prediction_ts",
        "Predicted_Score",
        "features",
        "model_version",
    }


def test_empty_unpacked_table_returns_zero_rows_with_correct_schema(spark):
    """empty_unpacked_table() must produce 0 rows and the 4 correct typed columns.

    This covers the fix for the first-run / no-traffic failure: when the
    _payload source is absent or empty, main() calls this helper to write a
    typed empty target table so the downstream setup_monitor task has a target.
    """
    from anime_score_predictor.monitoring.unpack_inference_table import empty_unpacked_table

    out = empty_unpacked_table(spark)

    assert out.count() == 0
    assert set(out.columns) == {
        "record_id",
        "prediction_ts",
        "Predicted_Score",
        "features",
        "model_version",
    }
    dtypes = dict(out.dtypes)
    assert dtypes["record_id"] == "string"
    assert dtypes["prediction_ts"] == "timestamp"
    assert dtypes["Predicted_Score"] == "double"
    assert dtypes["features"] == "array<double>"
    assert dtypes["model_version"] == "string"
