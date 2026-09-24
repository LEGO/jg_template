"""Feature Store Definitions and Table Configurations.

Compliant with Level 1 and Level 2 of LEGO AI Handbook Feature Store Maturity Framework:
https://baseplate.legogroup.io/docs/default/component/ds_ai_handbook/traditional_ml/docs/maturity_levels/8-mlops-feature-store/

Explanation:
Feature Store Level 1: Standalone feature table definitions decoupled from model training code,
persisted in a managed feature store (Databricks Feature Engineering / Unity Catalog / Feast).
Feature Store Level 2: Reusable modular feature components across models, automated extraction pipeline,
and live/online feature serving configuration.
"""

from typing import List, Optional
from dataclasses import dataclass, field
import yaml
from pathlib import Path

try:
    from databricks.feature_engineering import FeatureEngineeringClient
except ImportError:
    FeatureEngineeringClient = None  # type: ignore

try:
    import feast
except ImportError:
    feast = None  # type: ignore

from common.utils import get_logger

logger = get_logger()


@dataclass
class FeatureDefinition:
    """Represents a single feature in the feature store."""
    name: str
    data_type: str
    description: str


@dataclass
class FeatureTableDefinition:
    """Represents a feature table definition in the feature store."""
    name: str
    catalog: str
    schema: str
    primary_keys: List[str]
    timestamp_keys: Optional[List[str]] = None
    description: str = ""
    features: List[FeatureDefinition] = field(default_factory=list)
    online_serving_enabled: bool = True

    @property
    def fully_qualified_name(self) -> str:
        return f"{self.catalog}.{self.schema}.{self.name}"


def load_feature_store_config(config_path: Optional[Path] = None) -> FeatureTableDefinition:
    """Loads feature store configuration from YAML definition file."""
    if config_path is None:
        config_path = Path(__file__).parent / "feature_table_config.yml"

    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)["feature_store"]

    features = [
        FeatureDefinition(
            name=feat["name"],
            data_type=feat["type"],
            description=feat.get("description", "")
        )
        for feat in raw_config.get("features", [])
    ]

    return FeatureTableDefinition(
        name=raw_config["name"],
        catalog=raw_config["catalog"],
        schema=raw_config["schema"],
        primary_keys=raw_config["primary_keys"],
        timestamp_keys=raw_config.get("timestamp_keys"),
        description=raw_config.get("description", ""),
        features=features,
        online_serving_enabled=raw_config.get("online_store", {}).get("enabled", True),
    )


def register_feature_table(
    df,
    table_def: Optional[FeatureTableDefinition] = None,
    fe_client: Optional[object] = None,
) -> None:
    """Registers or updates a feature table in Databricks Feature Engineering / Unity Catalog."""
    if table_def is None:
        table_def = load_feature_store_config()

    logger.info(f"Registering feature table in Feature Store: {table_def.fully_qualified_name}")

    if fe_client is None and FeatureEngineeringClient is not None:
        fe_client = FeatureEngineeringClient()

    if fe_client is not None:
        try:
            fe_client.get_table(name=table_def.fully_qualified_name)
            logger.info(f"Feature table {table_def.fully_qualified_name} already exists. Writing features.")
            fe_client.write_table(
                name=table_def.fully_qualified_name,
                df=df,
                mode="merge",
            )
        except Exception:
            logger.info(f"Creating new feature table {table_def.fully_qualified_name} in Feature Store.")
            fe_client.create_table(
                name=table_def.fully_qualified_name,
                primary_keys=table_def.primary_keys,
                timestamp_keys=table_def.timestamp_keys,
                schema=df.schema,
                description=table_def.description,
            )
            fe_client.write_table(
                name=table_def.fully_qualified_name,
                df=df,
                mode="overwrite",
            )
