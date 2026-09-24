"""Feature Store module compliant with LEGO AI Handbook Level 2."""

from feature_store.feature_definitions import (
    FeatureTableDefinition,
    FeatureDefinition,
    load_feature_store_config,
    register_feature_table,
)
from feature_store.feature_pipeline import (
    FeatureExtractionPipeline,
    ModularFeatureExtractor,
)
from feature_store.online_serving import OnlineFeatureStoreManager

__all__ = [
    "FeatureTableDefinition",
    "FeatureDefinition",
    "load_feature_store_config",
    "register_feature_table",
    "FeatureExtractionPipeline",
    "ModularFeatureExtractor",
    "OnlineFeatureStoreManager",
]
