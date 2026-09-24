from unittest.mock import MagicMock

from databricks.sdk.errors import ResourceAlreadyExists

from lego_ml_product.monitoring import setup_monitor

TABLE = "cat.sch.preds_unpacked"
SCHEMA = "cat.sch"
ASSETS_DIR = "/Workspace/monitoring"
TABLE_ID = "7f38e280-130a-4ddb-84db-e96d43b5d32e"
SCHEMA_ID = "164b27e7-0d12-4ff7-ad7a-0f97ac606ee6"


def _client() -> MagicMock:
    client = MagicMock()
    client.tables.get.return_value = MagicMock(table_id=TABLE_ID)
    client.schemas.get.return_value = MagicMock(schema_id=SCHEMA_ID)
    return client


def _call(client: MagicMock) -> None:
    setup_monitor.create_or_update_monitor(
        workspace_client=client,
        table_name=TABLE,
        output_schema_name=SCHEMA,
        assets_dir=ASSETS_DIR,
    )


def test_creates_monitor_and_lets_databricks_refresh_it():
    client = _client()

    _call(client)

    client.data_quality.update_monitor.assert_not_called()
    # Databricks refreshes a newly created monitor itself.
    client.data_quality.create_refresh.assert_not_called()

    monitor = client.data_quality.create_monitor.call_args.kwargs["monitor"]
    # The API addresses objects by UUID, not name: the FQN fails with "UUID string too large".
    assert monitor.object_id == TABLE_ID
    config = monitor.data_profiling_config
    assert config.output_schema_id == SCHEMA_ID
    assert config.assets_dir == ASSETS_DIR
    assert config.inference_log.timestamp_column == "prediction_ts"
    assert config.inference_log.prediction_column == "Predicted_number_of_parts"


def test_updates_existing_monitor_and_requests_a_refresh():
    client = _client()
    client.data_quality.create_monitor.side_effect = ResourceAlreadyExists("exists")

    _call(client)

    kwargs = client.data_quality.update_monitor.call_args.kwargs
    assert kwargs["object_id"] == TABLE_ID
    assert kwargs["update_mask"] == "data_profiling_config"
    # An update changes config only, so metrics need an explicit recompute — requested
    # but never polled.
    client.data_quality.create_refresh.assert_called_once()
    client.data_quality.get_monitor.assert_not_called()
