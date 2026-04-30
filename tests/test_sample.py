import logging
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from common.utils import get_logger, load_model_config


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
    config_path = Path(__file__).parent.parent / "src/mlops_pipeline/model/model_config.yml"
    cfg = load_model_config(config_path=config_path)
    assert "tuning" in cfg
    assert not "auhwdukahwd" in cfg


def test_load_model_config_learning_rate():
    config_path = Path(__file__).parent.parent / "src/mlops_pipeline/model/model_config.yml"
    cfg = load_model_config(config_path=config_path)
    assert cfg.tuning.learning_rate.low > 0
    assert cfg.tuning.learning_rate.high <= 1
