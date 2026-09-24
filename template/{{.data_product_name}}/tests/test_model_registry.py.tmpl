"""Unit tests for MLflow Model Registry governance and promotion gates."""

from unittest.mock import MagicMock, patch
from common.model_registry import ModelRegistryManager


def test_register_model_to_registry():
    fake_client = MagicMock()
    fake_version = MagicMock(version="1")
    manager = ModelRegistryManager(client=fake_client)

    with patch("mlflow.register_model", return_value=fake_version):
        version = manager.register_model_to_registry(
            model_uri="runs:/123/model",
            registered_model_name="sandbox.lego_ml.lego_model",
            tags={"env": "dev"},
        )
        assert version.version == "1"
        fake_client.set_model_version_tag.assert_called()


def test_promote_to_champion():
    fake_client = MagicMock()
    manager = ModelRegistryManager(client=fake_client)

    manager.promote_to_champion("sandbox.lego_ml.lego_model", version="2")
    fake_client.set_registered_model_alias.assert_called_once_with(
        name="sandbox.lego_ml.lego_model",
        alias="champion",
        version="2",
    )


def test_rollback_model():
    fake_client = MagicMock()
    manager = ModelRegistryManager(client=fake_client)

    manager.rollback_model("sandbox.lego_ml.lego_model", fallback_version="1")
    fake_client.set_registered_model_alias.assert_called_once_with(
        name="sandbox.lego_ml.lego_model",
        alias="champion",
        version="1",
    )
