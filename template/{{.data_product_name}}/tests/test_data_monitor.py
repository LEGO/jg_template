"""Tests for the Lakehouse data-monitor setup (no Spark, mocked WorkspaceClient)."""

from unittest.mock import MagicMock

from databricks.sdk.errors import ResourceAlreadyExists

from data_monitoring.data_drift_report import setup_monitor


ASSETS_DIR = "/ai_agency/anime_score_predictor/dev/data_monitoring"


def test_creates_snapshot_monitor_when_absent():
    w = MagicMock()
    setup_monitor(w, "cat.sch.anime_features", "cat.sch", ASSETS_DIR)

    w.quality_monitors.create.assert_called_once()
    kwargs = w.quality_monitors.create.call_args.kwargs
    assert kwargs["table_name"] == "cat.sch.anime_features"
    assert kwargs["output_schema_name"] == "cat.sch"
    assert kwargs["assets_dir"] == ASSETS_DIR
    assert kwargs["snapshot"] is not None  # Snapshot profile (static table)
    w.quality_monitors.run_refresh.assert_not_called()


def test_refreshes_when_monitor_already_exists():
    w = MagicMock()
    w.quality_monitors.create.side_effect = ResourceAlreadyExists("exists")

    setup_monitor(w, "cat.sch.anime_features", "cat.sch", ASSETS_DIR)

    w.quality_monitors.run_refresh.assert_called_once_with(table_name="cat.sch.anime_features")
