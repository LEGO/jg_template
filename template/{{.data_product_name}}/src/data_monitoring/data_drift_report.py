"""Set up a Databricks Lakehouse data monitor on a Delta table.

Minimal wrapper over ``WorkspaceClient.quality_monitors``. Databricks Lakehouse
Monitoring natively profiles a Unity Catalog table — it computes descriptive /
data-quality statistics into a ``<table>_profile_metrics`` table, drift statistics
into ``<table>_drift_metrics``, and generates a monitoring dashboard. This replaces
any hand-rolled profiler.

Decoupled from any model, but tied to the ingestion step: by default it monitors
the anime feature table produced by ``anime_score_predictor``'s preprocessing job.

The anime feature table is a static snapshot (no event-time column), so we use a
Snapshot monitor: each refresh profiles the full table and drift is measured across
consecutive refreshes.
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
    return parser.parse_args()


def setup_monitor(
    workspace_client: WorkspaceClient,
    table_fqn: str,
    output_schema_name: str,
    assets_dir: str,
) -> None:
    """Creates a Snapshot data monitor on ``table_fqn``, or refreshes it if it exists.

    Args:
        workspace_client: Authenticated WorkspaceClient.
        table_fqn: Fully-qualified table to monitor (``catalog.schema.table``).
        output_schema_name: ``catalog.schema`` for the generated metric tables.
        assets_dir: Team-shared workspace folder for the monitor's generated assets.
    """
    try:
        workspace_client.quality_monitors.create(
            table_name=table_fqn,
            output_schema_name=output_schema_name,
            assets_dir=assets_dir,
            snapshot=MonitorSnapshot(),
        )
        logger.info("Created data monitor on %s (metrics -> %s).", table_fqn, output_schema_name)
    except ResourceAlreadyExists:
        logger.info("Monitor already exists on %s; triggering a refresh.", table_fqn)
        workspace_client.quality_monitors.run_refresh(table_name=table_fqn)


def main() -> None:
    """Entry point: create or refresh the data monitor for the configured table.

    This satisfied level 1 data monitoring of the MLOps Maturity Assessment framework: https://baseplate.legogroup.io/docs/default/component/ds_ai_handbook/traditional_ml/docs/maturity_levels/10-mlops-data-monitoring/
    explanation: Basic data monitoring is implemented to monitor key data properties.
    """
    args = _parse_args()

    table_fqn = f"{args.catalog_name}.{args.schema_name}.{args.table_name}"
    output_schema = args.output_schema_name

    workspace_client = WorkspaceClient(profile=args.profile) if args.profile else WorkspaceClient()
    setup_monitor(workspace_client, table_fqn, output_schema, args.assets_dir)


if __name__ == "__main__":
    main()
