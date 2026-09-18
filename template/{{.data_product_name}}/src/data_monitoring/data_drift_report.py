"""Set up a Databricks Lakehouse data monitor on a Delta table.

Minimal wrapper over ``WorkspaceClient.quality_monitors``. Databricks Lakehouse
Monitoring natively profiles a Unity Catalog table — it computes descriptive /
data-quality statistics into a ``<table>_profile_metrics`` table, drift statistics
into ``<table>_drift_metrics``, and generates a monitoring dashboard. This replaces
any hand-rolled profiler.

Decoupled from any model, but tied to the ingestion step: by default it monitors the
LEGO feature table produced by ``lego_parts_predictor``'s preprocessing job.

The LEGO feature table has no event-time column, so we use a Snapshot monitor: each
refresh profiles the full table, and drift is measured against the baseline table when
one is configured.
"""

import argparse

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import ResourceAlreadyExists
from databricks.sdk.service.catalog import MonitorSnapshot

from common.utils import get_logger

logger = get_logger()


def _parse_args() -> argparse.Namespace:
    """Parses CLI arguments for the monitor-setup job."""
    parser = argparse.ArgumentParser(description="Create/refresh a Databricks data monitor on a table.")
    parser.add_argument("--catalog_name", required=True, help="Catalog of the table to monitor.")
    parser.add_argument("--schema_name", required=True, help="Schema of the table to monitor.")
    parser.add_argument("--table_name", required=True, help="Table to monitor.")
    parser.add_argument(
        "--output_schema_name",
        required=True,
        help="Fully-qualified schema (catalog.schema) where metric tables are written.",
    )
    parser.add_argument(
        "--assets_dir",
        required=True,
        help="Workspace folder for the monitor's generated assets (e.g. its dashboard).",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Databricks CLI profile to authenticate with (omit when running on Databricks).",
    )
    parser.add_argument(
        "--baseline_table_name",
        required=False,
        default=None,
        help="Baseline table name for drift comparison (in the same catalog/schema).",
    )
    return parser.parse_args()


def setup_monitor(
    workspace_client: WorkspaceClient,
    table_fqn: str,
    output_schema_name: str,
    assets_dir: str,
    baseline_table_name: str | None = None,
) -> None:
    """Creates a Snapshot data monitor on ``table_fqn``, or refreshes it if it exists.

    Args:
        workspace_client: Authenticated WorkspaceClient.
        table_fqn: Fully-qualified table to monitor (``catalog.schema.table``).
        output_schema_name: ``catalog.schema`` for the generated metric tables.
        assets_dir: Team-shared workspace folder for the monitor's generated assets.
        baseline_table_name: Fully-qualified reference table. With a baseline, every
            refresh is compared against a fixed distribution instead of only against the
            previous refresh, so drift is monotonic and persistent.

    NOTE Lakehouse Monitoring profiles every column and the API offers no way to exclude
    one, so the primary key is monitored too. Because its values are unique by definition,
    its distribution "changes" on every refresh and the chi-squared test always reports
    significance (p ~ 1e-12) -- expect the id column to be flagged, and ignore it.
    """
    try:
        workspace_client.quality_monitors.create(
            table_name=table_fqn,
            output_schema_name=output_schema_name,
            assets_dir=assets_dir,
            snapshot=MonitorSnapshot(),
            baseline_table_name=baseline_table_name,
        )
        logger.info("Created data monitor on %s (metrics -> %s).", table_fqn, output_schema_name)
    except ResourceAlreadyExists:
        # Update rather than only refreshing: a monitor created before the baseline
        # table existed would otherwise never pick it up, and would silently report no
        # drift forever.
        logger.info("Monitor already exists on %s; updating it and triggering a refresh.", table_fqn)
        workspace_client.quality_monitors.update(
            table_name=table_fqn,
            output_schema_name=output_schema_name,
            snapshot=MonitorSnapshot(),
            baseline_table_name=baseline_table_name,
        )
        workspace_client.quality_monitors.run_refresh(table_name=table_fqn)


def main() -> None:
    """Entry point: create or refresh the data monitor for the configured table.

    This satisfied level 1 data monitoring of the MLOps Maturity Assessment framework: https://baseplate.legogroup.io/docs/default/component/ds_ai_handbook/traditional_ml/docs/maturity_levels/10-mlops-data-monitoring/
    explanation: Basic data monitoring is implemented to monitor key data properties.
    """
    args = _parse_args()

    table_fqn = f"{args.catalog_name}.{args.schema_name}.{args.table_name}"
    output_schema = args.output_schema_name

    baseline_fqn = (
        f"{args.catalog_name}.{args.schema_name}.{args.baseline_table_name}"
        if args.baseline_table_name
        else None
    )

    workspace_client = WorkspaceClient(profile=args.profile) if args.profile else WorkspaceClient()
    setup_monitor(workspace_client, table_fqn, output_schema, args.assets_dir, baseline_fqn)


if __name__ == "__main__":
    main()
