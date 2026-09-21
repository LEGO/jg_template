import argparse

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import ResourceAlreadyExists
from databricks.sdk.service.dataquality import (
    AggregationGranularity,
    DataProfilingConfig,
    InferenceLogConfig,
    InferenceProblemType,
    Monitor,
    Refresh,
)

from common.utils import get_logger

logger = get_logger()

_OBJECT_TYPE = "table"


def _data_profiling_config(output_schema_id: str, assets_dir: str) -> DataProfilingConfig:
    return DataProfilingConfig(
        output_schema_id=output_schema_id,
        assets_dir=assets_dir,
        inference_log=InferenceLogConfig(
            problem_type=InferenceProblemType.INFERENCE_PROBLEM_TYPE_REGRESSION,
            timestamp_column="prediction_ts",
            granularities=[AggregationGranularity.AGGREGATION_GRANULARITY_1_DAY],
            prediction_column="Predicted_number_of_parts",
            model_id_column="model_version",
        ),
    )


def create_or_update_monitor(
    workspace_client: WorkspaceClient,
    table_name: str,
    output_schema_name: str,
    assets_dir: str,
) -> None:
    """Create or update a data-quality monitor on ``table_name``, then request a refresh.

    Fire and forget: the refresh is requested and its state logged, but not waited on.
    Monitoring is best-effort, so a pending or failed refresh must not fail the job.

    Args:
        workspace_client: Databricks workspace client.
        table_name: Fully qualified unpacked predictions table (catalog.schema.table).
        output_schema_name: Fully qualified schema (catalog.schema) for monitor outputs.
        assets_dir: Workspace directory for monitor assets (dashboard, etc.).
    """
    # The data-quality API addresses objects by UUID, not by name.
    table_id = workspace_client.tables.get(full_name=table_name).table_id
    output_schema_id = workspace_client.schemas.get(full_name=output_schema_name).schema_id

    config = _data_profiling_config(output_schema_id=output_schema_id, assets_dir=assets_dir)

    try:
        workspace_client.data_quality.create_monitor(
            monitor=Monitor(
                object_type=_OBJECT_TYPE,
                object_id=table_id,
                data_profiling_config=config,
            )
        )
        logger.info(f"Created monitor for '{table_name}'")
    except ResourceAlreadyExists:
        workspace_client.data_quality.update_monitor(
            object_type=_OBJECT_TYPE,
            object_id=table_id,
            monitor=Monitor(
                object_type=_OBJECT_TYPE,
                object_id=table_id,
                data_profiling_config=config,
            ),
            update_mask="data_profiling_config",
        )
        logger.info(f"Updated monitor for '{table_name}'")

    refresh = workspace_client.data_quality.create_refresh(
        object_type=_OBJECT_TYPE,
        object_id=table_id,
        refresh=Refresh(object_type=_OBJECT_TYPE, object_id=table_id),
    )
    logger.info(f"Requested refresh for '{table_name}' (state: {refresh.state})")


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

    create_or_update_monitor(
        workspace_client=workspace_client,
        table_name=f"{args.catalog}.{args.schema}.{args.unpacked_table}",
        output_schema_name=f"{args.catalog}.{args.schema}",
        assets_dir=args.assets_dir,
    )
    logger.info("Monitor setup complete.")


if __name__ == "__main__":
    main()
