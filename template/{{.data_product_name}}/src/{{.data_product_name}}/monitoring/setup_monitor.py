import argparse
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.catalog import (
    MonitorInferenceLog,
    MonitorInferenceLogProblemType,
)

from common.utils import get_logger

logger = get_logger()

# Compared as plain strings: the SDK may hand back either a MonitorInfoStatus enum
# or a raw string, and these are the documented wire values.
_STATUS_ACTIVE = "MONITOR_STATUS_ACTIVE"
_STATUS_FAILURES = ("MONITOR_STATUS_ERROR", "MONITOR_STATUS_FAILED")


def _inference_log() -> MonitorInferenceLog:
    return MonitorInferenceLog(
        problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_REGRESSION,
        timestamp_col="prediction_ts",
        granularities=["1 day"],
        prediction_col="Predicted_number_of_parts",
        model_id_col="model_version",
    )


def _wait_until_active(
    workspace_client: WorkspaceClient,
    table_name: str,
    timeout_seconds: int = 1800,
    poll_seconds: int = 30,
) -> None:
    """Block until the monitor leaves MONITOR_STATUS_PENDING.

    Creating or updating a monitor returns as soon as the request is accepted; Databricks
    then provisions the metric tables and dashboard while the monitor sits in
    MONITOR_STATUS_PENDING. `run_refresh` is rejected in that state, so callers must wait.
    The SDK exposes no waiter for quality monitors, hence the explicit poll.

    Args:
        workspace_client: Databricks workspace client.
        table_name: Fully qualified monitored table (catalog.schema.table).
        timeout_seconds: Give up after this long so a stuck monitor fails the task
            instead of holding the cluster indefinitely.
        poll_seconds: Delay between status checks.

    Raises:
        RuntimeError: If the monitor reports a failure status.
        TimeoutError: If the monitor is still not active within `timeout_seconds`.
    """
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        info = workspace_client.quality_monitors.get(table_name=table_name)
        status = str(getattr(info.status, "value", info.status))

        if status == _STATUS_ACTIVE:
            logger.info(f"Monitor '{table_name}' is active")
            return
        if status in _STATUS_FAILURES:
            raise RuntimeError(f"Monitor for '{table_name}' entered {status}")

        logger.info(f"Monitor '{table_name}' is {status}; polling again in {poll_seconds}s")
        time.sleep(poll_seconds)

    raise TimeoutError(
        f"Monitor for '{table_name}' still not active after {timeout_seconds}s"
    )


def create_or_update_monitor(
    workspace_client: WorkspaceClient,
    table_name: str,
    output_schema_name: str,
    assets_dir: str,
) -> None:
    """Create or update a Lakehouse InferenceLog monitor, refreshing it when needed.

    A newly created monitor is refreshed automatically by Databricks, so an explicit
    refresh — and the wait for the monitor to leave PENDING that it requires — happens
    only on the update path. Safe to call repeatedly.

    Args:
        workspace_client: Databricks workspace client.
        table_name: Fully qualified unpacked predictions table (catalog.schema.table).
        output_schema_name: Fully qualified schema (catalog.schema) for monitor outputs.
        assets_dir: Workspace directory for monitor assets (dashboard, etc.).
    """
    try:
        workspace_client.quality_monitors.get(table_name=table_name)
        logger.info(f"Monitor for '{table_name}' exists. Updating...")
        workspace_client.quality_monitors.update(
            table_name=table_name,
            output_schema_name=output_schema_name,
            inference_log=_inference_log(),
        )
        # An update changes config only; metrics need an explicit recompute.
        needs_refresh = True
    except NotFound:
        logger.info(f"Monitor for '{table_name}' not found. Creating...")
        workspace_client.quality_monitors.create(
            table_name=table_name,
            output_schema_name=output_schema_name,
            assets_dir=assets_dir,
            inference_log=_inference_log(),
        )
        # Creation triggers its own initial refresh; a second one would be redundant.
        needs_refresh = False

    if needs_refresh:
        # Only the update path refreshes, so only it needs the monitor out of PENDING.
        _wait_until_active(workspace_client=workspace_client, table_name=table_name)
        logger.info(f"Triggering refresh for monitor '{table_name}'")
        workspace_client.quality_monitors.run_refresh(table_name=table_name)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create or update prediction drift monitor.")
    parser.add_argument("--catalog", required=True, help="Catalog name.")
    parser.add_argument("--schema", required=True, help="Schema name.")
    parser.add_argument("--unpacked_table", required=True, help="Unpacked predictions table name.")
    parser.add_argument("--assets_dir", required=True, help="Workspace dir for monitor assets.")
    return parser.parse_args()


def main() -> None:
    """Entry point: resolve names and create-or-update the drift monitor."""
    args = _parse_args()
    workspace_client = WorkspaceClient()

    table_name = f"{args.catalog}.{args.schema}.{args.unpacked_table}"
    output_schema_name = f"{args.catalog}.{args.schema}"

    create_or_update_monitor(
        workspace_client=workspace_client,
        table_name=table_name,
        output_schema_name=output_schema_name,
        assets_dir=args.assets_dir,
    )
    logger.info("Monitor setup complete.")


if __name__ == "__main__":
    main()
