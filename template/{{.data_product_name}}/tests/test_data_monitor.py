"""Tests for the Lakehouse data-monitor setup (no Spark, mocked WorkspaceClient)."""

from unittest.mock import MagicMock

from databricks.sdk.errors import ResourceAlreadyExists

from data_monitoring.data_drift_report import setup_monitor


ASSETS_DIR = "/ai_agency/lego_parts_predictor/dev/data_monitoring"
TABLE = "cat.sch.lego_set_features"
BASELINE = "cat.sch.lego_set_features_baseline"


def test_creates_snapshot_monitor_when_absent():
    w = MagicMock()
    setup_monitor(w, TABLE, "cat.sch", ASSETS_DIR)

    w.quality_monitors.create.assert_called_once()
    kwargs = w.quality_monitors.create.call_args.kwargs
    assert kwargs["table_name"] == TABLE
    assert kwargs["output_schema_name"] == "cat.sch"
    assert kwargs["assets_dir"] == ASSETS_DIR
    assert kwargs["snapshot"] is not None  # Snapshot profile (static table)
    w.quality_monitors.run_refresh.assert_not_called()


def test_passes_baseline_table_when_given():
    w = MagicMock()
    setup_monitor(w, TABLE, "cat.sch", ASSETS_DIR, baseline_table_name=BASELINE)

    assert w.quality_monitors.create.call_args.kwargs["baseline_table_name"] == BASELINE


def test_updates_existing_monitor_so_the_baseline_is_attached():
    # A monitor created before the baseline table existed would otherwise keep no
    # baseline forever, silently reporting no drift.
    w = MagicMock()
    w.quality_monitors.create.side_effect = ResourceAlreadyExists("exists")

    setup_monitor(w, TABLE, "cat.sch", ASSETS_DIR, baseline_table_name=BASELINE)

    w.quality_monitors.update.assert_called_once()
    assert w.quality_monitors.update.call_args.kwargs["baseline_table_name"] == BASELINE
    w.quality_monitors.run_refresh.assert_called_once_with(table_name=TABLE)
