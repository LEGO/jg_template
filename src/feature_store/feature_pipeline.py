"""Dedicated Feature Extraction Pipeline and Modular Feature Components.

Compliant with Level 2 of LEGO AI Handbook Feature Store Maturity Framework:
https://baseplate.legogroup.io/docs/default/component/ds_ai_handbook/traditional_ml/docs/maturity_levels/8-mlops-feature-store/

Explanation:
Feature Store Level 2: Dedicated feature extraction pipeline modules that run independently
of model training, producing reusable modular feature components across multiple models,
and supporting automated materialization to both offline and online feature stores.
"""

from typing import Dict, Any, List
from pyspark.sql import DataFrame, SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType, DoubleType

from databricks.feature_engineering import FeatureEngineeringClient
from common.utils import get_logger
from feature_store.feature_definitions import load_feature_store_config, register_feature_table

logger = get_logger()


class ModularFeatureExtractor:
    """Reusable feature transformation and extraction components across models."""

    @staticmethod
    def extract_temporal_features(df: DataFrame, year_col: str = "year") -> DataFrame:
        """Extracts age and era temporal features."""
        current_year = 2026
        return df.withColumn("set_age_years", F.lit(current_year) - F.col(year_col))

    @staticmethod
    def extract_ratio_features(df: DataFrame, piece_col: str = "pieces", price_col: str = "price") -> DataFrame:
        """Extracts piece-to-price ratio features."""
        return df.withColumn(
            "price_per_piece",
            F.when(F.col(piece_col) > 0, F.col(price_col) / F.col(piece_col)).otherwise(0.0),
        )

    @staticmethod
    def extract_categorical_features(df: DataFrame, category_col: str = "theme_group") -> DataFrame:
        """Encodes and indexes categorical features."""
        return df.withColumn(
            f"{category_col}_clean",
            F.lower(F.trim(F.coalesce(F.col(category_col), F.lit("unknown")))),
        )


class FeatureExtractionPipeline:
    """Dedicated feature extraction pipeline running independently of model training."""

    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.fe_client = FeatureEngineeringClient()
        self.config = load_feature_store_config()

    def run_pipeline(self, raw_df: DataFrame) -> DataFrame:
        """Executes modular feature extraction and writes to feature store."""
        logger.info("Executing modular feature extraction pipeline...")

        # Apply modular feature extractors
        features_df = ModularFeatureExtractor.extract_temporal_features(raw_df)
        features_df = ModularFeatureExtractor.extract_ratio_features(features_df)
        features_df = ModularFeatureExtractor.extract_categorical_features(features_df)

        # Register in Feature Store
        register_feature_table(features_df, table_def=self.config, fe_client=self.fe_client)
        logger.info(f"Feature extraction pipeline complete. Stored in {self.config.fully_qualified_name}")
        return features_df
