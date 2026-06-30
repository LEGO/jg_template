from unittest.mock import MagicMock, patch

import pytest


def test_module_imports_without_instantiating_client():
    """Importing common.spark_helper must not construct a FeatureEngineeringClient
    at module load time — otherwise importing this module outside a Databricks
    runtime crashes the process.
    """
    with patch("databricks.feature_engineering.FeatureEngineeringClient") as ctor:
        import importlib

        import common.spark_helper as sh

        importlib.reload(sh)
        ctor.assert_not_called()


def test_register_short_circuits_when_table_already_registered():
    """If fe.get_table(...) succeeds, no SQL is run and fe.create_table is not called."""
    import common.spark_helper as sh

    fake_fe = MagicMock()
    fake_fe.get_table.return_value = MagicMock()  # table found
    fake_spark = MagicMock()

    with patch.object(sh, "FeatureEngineeringClient", return_value=fake_fe):
        sh.register_delta_table_in_feature_store(
            spark=fake_spark,
            fully_qualified_path="cat.sch.tbl",
            primary_keys=["id"],
        )

    fake_fe.get_table.assert_called_once_with(name="cat.sch.tbl")
    fake_fe.create_table.assert_not_called()
    fake_spark.sql.assert_not_called()


def test_register_creates_table_when_get_table_raises_value_error():
    """If fe.get_table(...) raises ValueError, apply PK constraints and call fe.create_table."""
    import common.spark_helper as sh

    fake_fe = MagicMock()
    fake_fe.get_table.side_effect = ValueError("not found")
    fake_spark = MagicMock()

    with patch.object(sh, "FeatureEngineeringClient", return_value=fake_fe):
        sh.register_delta_table_in_feature_store(
            spark=fake_spark,
            fully_qualified_path="cat.sch.tbl",
            primary_keys=["id"],
            description="desc",
            tags={"team": "ml"},
        )

    # Two ALTER TABLE statements per primary key: NOT NULL, then ADD CONSTRAINT.
    sql_calls = [call.args[0] for call in fake_spark.sql.call_args_list]
    assert any("ALTER COLUMN id SET NOT NULL" in s for s in sql_calls)
    assert any("ADD CONSTRAINT tbl_pk PRIMARY KEY(id)" in s for s in sql_calls)

    fake_fe.create_table.assert_called_once_with(
        name="cat.sch.tbl",
        primary_keys=["id"],
        source="cat.sch.tbl",
        description="desc",
        tags={"team": "ml"},
    )


def test_register_propagates_non_value_error_exceptions():
    """Bare `except:` was masking real failures. Non-ValueError exceptions must propagate."""
    import common.spark_helper as sh

    fake_fe = MagicMock()
    fake_fe.get_table.side_effect = RuntimeError("permission denied")
    fake_spark = MagicMock()

    with patch.object(sh, "FeatureEngineeringClient", return_value=fake_fe):
        with pytest.raises(RuntimeError):
            sh.register_delta_table_in_feature_store(
                spark=fake_spark,
                fully_qualified_path="cat.sch.tbl",
                primary_keys=["id"],
            )
