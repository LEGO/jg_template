import argparse

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.catalog import (
    MonitorInferenceLog,
    MonitorInferenceLogProblemType,
)

from common.utils import get_logger

logger = get_logger()


def _inference_log() -> MonitorInferenceLog:
    return MonitorInferenceLog(
        problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_REGRESSION,
        timestamp_col="prediction_ts",
        granularities=["1 day"],
        prediction_col="Predicted_Score",
        model_id_col="model_version",
    )


def create_or_update_monitor(
    workspace_client: WorkspaceClient,
    table_name: str,
    output_schema_name: str,
    assets_dir: str,
) -> None:
    """Create or update a Lakehouse InferenceLog monitor, then trigger a refresh.

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
    except NotFound:
        logger.info(f"Monitor for '{table_name}' not found. Creating...")
        workspace_client.quality_monitors.create(
            table_name=table_name,
            output_schema_name=output_schema_name,
            assets_dir=assets_dir,
            inference_log=_inference_log(),
        )

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
