from unittest.mock import MagicMock, patch

import common.mlflow_helper as mh


def test_started_run_is_tagged_with_template_provenance():
    """Every run started via the helper must carry the template provenance tag,
    so runs created by a generated project are traceable back to the template.
    """
    with (
        patch.object(mh.mlflow, "set_registry_uri"),
        patch.object(mh.mlflow, "get_experiment_by_name", return_value=MagicMock()),
        patch.object(mh.mlflow, "set_experiment"),
        patch.object(mh.mlflow, "end_run"),
        patch.object(mh.mlflow, "start_run", return_value=MagicMock()) as start_run,
        patch.object(mh.mlflow, "set_tag") as set_tag,
    ):
        run = mh.start_mlflow_experiment_and_run("/some/experiment")

    assert run is start_run.return_value
    set_tag.assert_called_once_with(mh.TEMPLATE_TAG_KEY, mh.TEMPLATE_TAG_VALUE)


def test_tag_model_as_template_generated_tags_model_and_version():
    fake_client = MagicMock()

    with patch.object(mh.mlflow, "MlflowClient", return_value=fake_client):
        mh.tag_model_as_template_generated("cat.sch.model", version=3)

    fake_client.set_registered_model_tag.assert_called_once_with(
        name="cat.sch.model",
        key=mh.TEMPLATE_TAG_KEY,
        value=mh.TEMPLATE_TAG_VALUE,
    )
    fake_client.set_model_version_tag.assert_called_once_with(
        name="cat.sch.model",
        version="3",
        key=mh.TEMPLATE_TAG_KEY,
        value=mh.TEMPLATE_TAG_VALUE,
    )


def test_tagging_failure_does_not_raise():
    """A tagging failure must never fail the training job."""
    fake_client = MagicMock()
    fake_client.set_registered_model_tag.side_effect = RuntimeError("no permission")

    with patch.object(mh.mlflow, "MlflowClient", return_value=fake_client):
        mh.tag_model_as_template_generated("cat.sch.model", version=1)


def test_champion_alias_also_applies_template_tag():
    fake_client = MagicMock()
    fake_client.search_model_versions.return_value = [
        MagicMock(version="1"),
        MagicMock(version="2"),
    ]

    with patch.object(mh.mlflow, "MlflowClient", return_value=fake_client):
        mh.set_champion_alias_on_logged_model("cat.sch.model")

    fake_client.set_registered_model_alias.assert_called_once_with(
        name="cat.sch.model",
        alias="champion",
        version="2",
    )
    fake_client.set_model_version_tag.assert_called_once_with(
        name="cat.sch.model",
        version="2",
        key=mh.TEMPLATE_TAG_KEY,
        value=mh.TEMPLATE_TAG_VALUE,
    )
