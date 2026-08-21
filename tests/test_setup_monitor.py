from unittest.mock import MagicMock

from databricks.sdk.errors import NotFound


def _client():
    client = MagicMock()
    return client


def test_creates_monitor_when_not_found():
    from anime_score_predictor.monitoring.setup_monitor import create_or_update_monitor

    client = _client()
    client.quality_monitors.get.side_effect = NotFound("nope")

    create_or_update_monitor(
        workspace_client=client,
        table_name="cat.sch.preds",
        output_schema_name="cat.sch",
        assets_dir="/Workspace/monitoring",
    )

    client.quality_monitors.create.assert_called_once()
    client.quality_monitors.update.assert_not_called()
    client.quality_monitors.run_refresh.assert_called_once_with(table_name="cat.sch.preds")

    kwargs = client.quality_monitors.create.call_args.kwargs
    assert kwargs["table_name"] == "cat.sch.preds"
    assert kwargs["output_schema_name"] == "cat.sch"
    assert kwargs["assets_dir"] == "/Workspace/monitoring"
    assert kwargs["inference_log"].timestamp_col == "prediction_ts"
    assert kwargs["inference_log"].prediction_col == "Predicted_Score"
    assert kwargs["inference_log"].model_id_col == "model_version"
    assert kwargs["inference_log"].granularities == ["1 day"]


def test_updates_monitor_when_exists():
    from anime_score_predictor.monitoring.setup_monitor import create_or_update_monitor

    client = _client()
    client.quality_monitors.get.return_value = MagicMock()  # exists

    create_or_update_monitor(
        workspace_client=client,
        table_name="cat.sch.preds",
        output_schema_name="cat.sch",
        assets_dir="/Workspace/monitoring",
    )

    client.quality_monitors.update.assert_called_once()
    client.quality_monitors.create.assert_not_called()
    client.quality_monitors.run_refresh.assert_called_once_with(table_name="cat.sch.preds")
