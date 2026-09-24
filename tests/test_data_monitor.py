"""Tests for the Lakehouse data-monitor setup (no Spark, mocked WorkspaceClient)."""

from unittest.mock import MagicMock

from databricks.sdk.errors import ResourceAlreadyExists

from data_monitoring.data_drift_report import setup_monitor


ASSETS_DIR = "/ai_agency/lego_parts_predictor/dev/data_monitoring"
TABLE = "cat.sch.lego_set_features"
SCHEMA = "cat.sch"
BASELINE = "cat.sch.lego_set_features_baseline"
TABLE_ID = "7f38e280-130a-4ddb-84db-e96d43b5d32e"
SCHEMA_ID = "164b27e7-0d12-4ff7-ad7a-0f97ac606ee6"


def _client() -> MagicMock:
    client = MagicMock()
    client.tables.get.return_value = MagicMock(table_id=TABLE_ID)
    client.schemas.get.return_value = MagicMock(schema_id=SCHEMA_ID)
    return client


def test_creates_snapshot_monitor_when_absent():
    client = _client()

    setup_monitor(client, TABLE, SCHEMA, ASSETS_DIR)

    client.data_quality.create_monitor.assert_called_once()
    client.data_quality.update_monitor.assert_not_called()

    monitor = client.data_quality.create_monitor.call_args.kwargs["monitor"]
    assert monitor.object_type == "table"
    # The API addresses objects by UUID, not name: the FQN fails with "UUID string too large".
    assert monitor.object_id == TABLE_ID
    config = monitor.data_profiling_config
    assert config.output_schema_id == SCHEMA_ID
    assert config.assets_dir == ASSETS_DIR
    assert config.snapshot is not None  # Snapshot profile (static table, no event time)


def test_passes_baseline_table_when_given():
    client = _client()

    setup_monitor(client, TABLE, SCHEMA, ASSETS_DIR, baseline_table_name=BASELINE)

    config = client.data_quality.create_monitor.call_args.kwargs["monitor"].data_profiling_config
    assert config.baseline_table_name == BASELINE


def test_updates_existing_monitor_so_the_baseline_is_attached():
    # A monitor created before the baseline table existed would otherwise keep no
    # baseline forever, silently reporting no drift.
    client = _client()
    client.data_quality.create_monitor.side_effect = ResourceAlreadyExists("exists")

    setup_monitor(client, TABLE, SCHEMA, ASSETS_DIR, baseline_table_name=BASELINE)

    client.data_quality.update_monitor.assert_called_once()
    kwargs = client.data_quality.update_monitor.call_args.kwargs
    assert kwargs["object_id"] == TABLE_ID
    assert kwargs["update_mask"] == "data_profiling_config"
    assert kwargs["monitor"].data_profiling_config.baseline_table_name == BASELINE


def test_create_does_not_request_a_redundant_refresh():
    """Databricks refreshes a newly created monitor itself."""
    client = _client()

    setup_monitor(client, TABLE, SCHEMA, ASSETS_DIR)

    client.data_quality.create_refresh.assert_not_called()


def test_update_requests_a_refresh_without_waiting():
    """An update changes config only, so metrics need an explicit recompute."""
    client = _client()
    client.data_quality.create_monitor.side_effect = ResourceAlreadyExists("exists")

    setup_monitor(client, TABLE, SCHEMA, ASSETS_DIR)

    client.data_quality.create_refresh.assert_called_once()
    assert client.data_quality.create_refresh.call_args.kwargs["object_id"] == TABLE_ID
    client.data_quality.get_monitor.assert_not_called()
