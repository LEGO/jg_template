"""Unit tests for Feature Store definitions, configuration, and extractors."""

from unittest.mock import MagicMock, patch
from feature_store.feature_definitions import (
    load_feature_store_config,
    register_feature_table,
    FeatureTableDefinition,
)
from feature_store.feature_pipeline import ModularFeatureExtractor
from feature_store.online_serving import OnlineFeatureStoreManager


def test_load_feature_store_config():
    config = load_feature_store_config()
    assert config.name == "lego_set_features"
    assert "set_id" in config.primary_keys
    assert len(config.features) > 0


def test_register_feature_table_existing():
    fake_client = MagicMock()
    fake_df = MagicMock()
    table_def = FeatureTableDefinition(
        name="test_table",
        catalog="cat",
        schema="sch",
        primary_keys=["id"],
    )
    register_feature_table(fake_df, table_def=table_def, fe_client=fake_client)
    fake_client.write_table.assert_called_once()


def test_online_feature_store_lookup():
    manager = OnlineFeatureStoreManager(workspace_client=MagicMock(), fe_client=MagicMock())
    entities = [{"set_id": "75192"}]
    features = manager.lookup_features_for_serving(entities)
    assert len(features) == 1
    assert "pieces" in features[0]
