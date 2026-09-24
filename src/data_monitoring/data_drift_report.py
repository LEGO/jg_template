"""Set up a Databricks Lakehouse data monitor on a Delta table.

Minimal wrapper over ``WorkspaceClient.data_quality``. Databricks Lakehouse
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
from databricks.sdk.service.dataquality import (
    DataProfilingConfig,
    Monitor,
    Refresh,
    SnapshotConfig,
)

from common.utils import get_logger

logger = get_logger()

_OBJECT_TYPE = "table"


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
    """Creates a Snapshot data monitor on ``table_fqn``, or updates it if it exists.

    A newly created monitor is refreshed by Databricks, so only the update path requests
    one. Fire and forget: the refresh state is logged, never waited on — monitoring is
    best-effort and a pending or failed refresh must not fail the job.

    Args:
        workspace_client: Authenticated WorkspaceClient.
        table_fqn: Fully-qualified table to monitor (``catalog.schema.table``).
        output_schema_name: ``catalog.schema`` for the generated metric tables.
        assets_dir: Team-shared workspace folder for the monitor's generated assets.
        baseline_table_name: Fully-qualified reference table. With a baseline, every
            refresh is compared against a fixed distribution instead of only against the
            previous refresh, so drift is monotonic and persistent.
    """
    # The data-quality API addresses objects by UUID, not by name.
    table_id = workspace_client.tables.get(full_name=table_fqn).table_id
    output_schema_id = workspace_client.schemas.get(full_name=output_schema_name).schema_id

    config = DataProfilingConfig(
        output_schema_id=output_schema_id,
        assets_dir=assets_dir,
        baseline_table_name=baseline_table_name,
        snapshot=SnapshotConfig(),
    )
    monitor = Monitor(
        object_type=_OBJECT_TYPE, object_id=table_id, data_profiling_config=config
    )

    try:
        workspace_client.data_quality.create_monitor(monitor=monitor)
        # Databricks refreshes a newly created monitor itself.
        logger.info("Created data monitor on %s (metrics -> %s).", table_fqn, output_schema_name)
        return
    except ResourceAlreadyExists:
        workspace_client.data_quality.update_monitor(
            object_type=_OBJECT_TYPE,
            object_id=table_id,
            monitor=monitor,
            update_mask="data_profiling_config",
        )
        logger.info("Updated data monitor on %s.", table_fqn)

    # An update changes config only; metrics need an explicit recompute, and no schedule
    # is set on the monitor so nothing else triggers one.
    refresh = workspace_client.data_quality.create_refresh(
        object_type=_OBJECT_TYPE,
        object_id=table_id,
        refresh=Refresh(object_type=_OBJECT_TYPE, object_id=table_id),
    )
    logger.info("Requested refresh for %s (state: %s).", table_fqn, refresh.state)


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
