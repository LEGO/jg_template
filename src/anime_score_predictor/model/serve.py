from enum import Enum
from common.utils import get_logger
import argparse

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    AiGatewayInferenceTableConfig,
    EndpointCoreConfigInput,
    ServedEntityInput,
    ServedModelInputWorkloadType,
    ServingEndpointAccessControlRequest,
    ServingEndpointPermissionLevel,
)


class RunStatusValues(str, Enum):
    """Enum for MLflow run status values."""

    RUNNING = "RUNNING"
    FAILED = "FAILED"
    FINISHED = "FINISHED"


logger = get_logger()

'''
Model serving script compliant with level 2 of https://baseplate.legogroup.io/catalog/default/component/ds_ai_handbook/docs/traditional_ml/docs/maturity_levels/4-mlops-model-serving/#model-serving. 

Explanation: 
Model Serving 1) Making predictions available for serving is automatic and new versions of a model are automatically made available for serving by using the model registry with a "champion" alias. 
'''


def _parse_args() -> argparse.Namespace:
    """Parses and returns CLI arguments for the ingestion pipeline.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run the model training pipeline.",
    )
    parser.add_argument(
        "--catalog",
        required=True,
        help="Catalog name where the training table will be written.",
    )
    parser.add_argument(
        "--schema",
        required=True,
        help="Schema (database) name where the training table will be written.",
    )
    parser.add_argument(
        "--model_name",
        required=True,
        help="Name of the model to be trained.",
    )
    parser.add_argument(
        "--model_alias",
        required=False,
        default="champion",
        help="Model alias to point to the logged model version.",
    )
    parser.add_argument(
        "--serving_endpoint_name",
        required=True,
        help="Name of the Databricks Model Serving endpoint to create or update.",
    )
    parser.add_argument(
        "--permission_group",
        required=True,
        help="Databricks permission group to grant CAN_MANAGE permission on the endpoint.",
    )
    parser.add_argument(
        "--workload_size",
        required=False,
        default="Small",
    )
    parser.add_argument(
        "--scale_to_zero",
        type=bool,
        default=True,
        help="Whether to scale to zero when idle (default: True)",
    )
    return parser.parse_args()


def get_model_version(
    workspace_client: WorkspaceClient,
    full_model_name: str,
    alias: str,
) -> str:
    """Gets the model version for a given alias.

    Args:
        workspace_client (WorkspaceClient): Databricks workspace client.
        full_model_name (str): Full Unity Catalog model name (catalog.schema.model).
        alias (str): Model alias (e.g., 'champion', 'production').

    Returns:
        str: Model version string.
    """
    logger.info(f"Getting model version for {full_model_name}@{alias}")
    model_version_info = workspace_client.model_versions.get_by_alias(
        full_name=full_model_name,
        alias=alias,
    )
    logger.info(f"Found model version: {model_version_info.version}")
    return str(model_version_info.version)


def create_or_update_endpoint(
    workspace_client: WorkspaceClient,
    endpoint_name: str,
    full_model_name: str,
    model_version: str,
    catalog: str,
    schema: str,
    workload_size: str = "Small",
    scale_to_zero: bool = True,
) -> None:
    """Creates or updates a Databricks Model Serving endpoint.

    Args:
        workspace_client (WorkspaceClient): Databricks workspace client.
        endpoint_name (str): Name of the serving endpoint.
        full_model_name (str): Full Unity Catalog model name.
        model_version (str): Model version to deploy.
        catalog (str): Unity Catalog name.
        schema (str): Unity Catalog schema name.
        workload_size (str): Size of the serving workload (default: "Small").
        scale_to_zero (bool): Whether to scale to zero when idle (default: True).
    """
    served_entity = ServedEntityInput(
        entity_name=full_model_name,
        entity_version=model_version,
        workload_size=workload_size,
        workload_type=ServedModelInputWorkloadType.CPU,
        scale_to_zero_enabled=scale_to_zero,
    )

    # AI Gateway inference table config. Legacy auto_capture_config is deprecated;
    # AI Gateway is now the supported path and still produces <prefix>_payload.
    inference_table_config = AiGatewayInferenceTableConfig(
        catalog_name=catalog,
        schema_name=schema,
        table_name_prefix=endpoint_name,   # -> <prefix>_payload
        enabled=True,
    )

    try:
        # Check if endpoint exists
        workspace_client.serving_endpoints.get(name=endpoint_name)
        logger.info(f"Endpoint '{endpoint_name}' exists. Updating configuration...")

        # Update the endpoint configuration
        workspace_client.serving_endpoints.update_config_and_wait(
            name=endpoint_name,
            served_entities=[served_entity],
        )
        logger.info(f"Endpoint '{endpoint_name}' updated successfully.")

    except Exception as e:
        if "RESOURCE_DOES_NOT_EXIST" in str(e) or "does not exist" in str(e).lower():
            logger.info(f"Endpoint '{endpoint_name}' does not exist. Creating...")

            workspace_client.serving_endpoints.create_and_wait(
                name=endpoint_name,
                config=EndpointCoreConfigInput(
                    name=endpoint_name,
                    served_entities=[served_entity],
                ),
            )
            logger.info(f"Endpoint '{endpoint_name}' created successfully.")
        else:
            logger.error(f"Error managing endpoint: {e}")
            raise

    # Configure the AI Gateway inference table (create or update; idempotent).
    logger.info(f"Configuring AI Gateway inference table for '{endpoint_name}'...")
    workspace_client.serving_endpoints.put_ai_gateway(
        name=endpoint_name,
        inference_table_config=inference_table_config,
    )
    logger.info(f"AI Gateway inference table configured for '{endpoint_name}'.")


def set_endpoint_permissions(
    workspace_client: WorkspaceClient,
    endpoint_name: str,
    permission_group: str,
) -> None:
    """Sets permissions on a serving endpoint.

    Args:
        workspace_client (WorkspaceClient): Databricks workspace client.
        endpoint_name (str): Name of the serving endpoint.
        permission_group (str): Group to grant CAN_MANAGE permission.
    """
    serving_endpoint = workspace_client.serving_endpoints.get(name=endpoint_name)

    workspace_client.serving_endpoints.update_permissions(
        serving_endpoint_id=serving_endpoint.id,
        access_control_list=[
            ServingEndpointAccessControlRequest(
                group_name=permission_group,
                permission_level=ServingEndpointPermissionLevel.CAN_MANAGE,
            )
        ],
    )
    logger.info(f"Permissions updated for endpoint '{endpoint_name}'")


def main() -> None:
    """Main entry point for serving endpoint deployment.

    Orchestrates the deployment process:
    1. Parses arguments.
    2. Initializes Databricks workspace client.
    3. Resolves the model version (champion alias).
    4. Creates or updates the serving endpoint.
    5. Sets permissions for the specified group.
    """
    args = _parse_args()

    # Use Unity Catalog-backed MLflow registry
    mlflow.set_registry_uri("databricks-uc")

    # Initialize workspace client
    workspace_client = WorkspaceClient()

    # Construct full model name
    full_model_name = f"{args.catalog}.{args.schema}.{args.model_name}"
    logger.info(f"Deploying model: {full_model_name}")

    try:
        # Get model version from alias
        model_version = get_model_version(
            workspace_client = workspace_client,
            full_model_name=full_model_name,
            alias=args.model_alias,
        )

        # Create or update the endpoint
        create_or_update_endpoint(
            workspace_client = workspace_client,
            endpoint_name = args.serving_endpoint_name,
            full_model_name=full_model_name,
            model_version=model_version,
            catalog=args.catalog,
            schema=args.schema,
            workload_size=args.workload_size,
            scale_to_zero=args.scale_to_zero,
        )

        # Set permissions
        set_endpoint_permissions(
            workspace_client = workspace_client,
            endpoint_name = args.serving_endpoint_name,
            permission_group = args.permission_group,
        )

        # Log endpoint URL
        endpoint_url = f"https://{workspace_client.config.host}/serving-endpoints/{args.serving_endpoint_name}/invocations"

        logger.info("Deployment complete!")
        logger.info(f"Endpoint URL: {endpoint_url}")
        logger.info("")
        logger.info("Example request:")
        logger.info(f'  curl -X POST "{endpoint_url}" \\')
        logger.info('    -H "Authorization: Bearer $DATABRICKS_TOKEN" \\')
        logger.info('    -H "Content-Type: application/json" \\')
        logger.info('    -d \'{"dataframe_records": [{"feature1": 1.0, "feature2": 2.0}]}\'')

    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        raise


if __name__ == "__main__":
    main()
