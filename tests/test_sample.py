import logging
from pathlib import Path

from common.utils import get_logger, load_model_config

"""
The test suite is (arguably) compliant with level 2 of https://baseplate.legogroup.io/catalog/default/component/ds_ai_handbook/docs/traditional_ml/docs/maturity_levels/9-mlops-continuous-integration/#continuous-integration. 

Explanation: 
Continuous Integration 1) Unit tests are designed to test individual components of the code in isolation, ensuring that each function or class behaves as expected. 
Continuous Integration 2) Integration test is effectively done on the NEXUS dev environment. Also in the deployment pipeline we validate the bundle. 

Caveat: this test suite is very narrow by design as it serves as an example. For business critical pipelines, you would want to expand this test suite to cover more edge cases and potential failure points.
Furthermore proper integration tests are missing in this version. 
"""


def test_get_logger_returns_logger_instance():
    logger = get_logger()
    assert isinstance(logger, logging.Logger)


def test_get_logger_name_matches_module():
    logger = get_logger()
    assert logger.name == "common.utils"


def test_get_logger_suppresses_noisy_libraries():
    get_logger()
    assert logging.getLogger("py4j").level == logging.WARN
    assert logging.getLogger("pyspark").level == logging.ERROR
    assert logging.getLogger("urllib3").level == logging.WARN
    assert logging.getLogger("tensorflow").level == logging.ERROR


def test_load_model_config_has_required_keys():
    config_path = Path(__file__).parent.parent / "src/lego_ml_product/model/model_config.yml"
    cfg = load_model_config(config_path=config_path)
    assert "tuning" in cfg
    assert not "auhwdukahwd" in cfg


def test_load_model_config_learning_rate():
    config_path = Path(__file__).parent.parent / "src/lego_ml_product/model/model_config.yml"
    cfg = load_model_config(config_path=config_path)
    assert cfg.tuning.learning_rate.low > 0
    assert cfg.tuning.learning_rate.high <= 1
