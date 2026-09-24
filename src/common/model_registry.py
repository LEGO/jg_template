"""MLflow Model Registry Management, Promotion Gates, and Governance.

Compliant with Level 1 and Level 2 of LEGO AI Handbook Model Registry Maturity Framework:
https://baseplate.legogroup.io/docs/default/component/ds_ai_handbook/traditional_ml/docs/maturity_levels/7-mlops-model-registry/

Explanation:
Model Registry Level 1: Models are tracked in a centralized registry along with versioning,
metadata, environment tags, and lineages using mlflow.register_model.
Model Registry Level 2: Deployment governance with automated evaluation gates, model promotion stages
(Staging -> Production / @challenger -> @champion), approval workflows, and instant rollback capabilities.
"""

from typing import Dict, Any, Optional
import mlflow
from mlflow.tracking import MlflowClient
from common.utils import get_logger

logger = get_logger()


class ModelRegistryManager:
    """Manages model registration, versioning, governance, and promotion gates."""

    def __init__(self, client: Optional[MlflowClient] = None):
        self.client = client or MlflowClient()

    def register_model_to_registry(
        self,
        model_uri: str,
        registered_model_name: str,
        tags: Optional[Dict[str, str]] = None,
        description: Optional[str] = None,
    ) -> Any:
        """Level 1: Registers a trained model to the MLflow Model Registry with metadata and lineage.

        Args:
            model_uri: URI of the logged model (e.g., 'runs:/<run_id>/model')
            registered_model_name: Fully qualified model name in Unity Catalog or MLflow Registry
            tags: Metadata tags (e.g., source_template, framework, data_product)
            description: Model documentation and provenance description

        Returns:
            ModelVersion: The registered model version object.
        """
        logger.info(f"Registering model to MLflow Model Registry: {registered_model_name} from {model_uri}")
        
        # Explicit call to mlflow.register_model for Level 1 compliance
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=registered_model_name,
        )

        default_tags = {
            "source_template": "mlops_template",
            "framework": "scikit-learn",
            "governance_status": "registered",
        }
        if tags:
            default_tags.update(tags)

        for key, val in default_tags.items():
            self.client.set_model_version_tag(
                name=registered_model_name,
                version=model_version.version,
                key=key,
                value=str(val),
            )

        if description:
            self.client.update_model_version(
                name=registered_model_name,
                version=model_version.version,
                description=description,
            )

        logger.info(
            f"Successfully registered model {registered_model_name} version {model_version.version}"
        )
        return model_version

    def evaluate_and_promote_model(
        self,
        registered_model_name: str,
        challenger_version: str,
        metric_name: str = "rmse",
        metric_threshold: float = 0.0,
        lower_is_better: bool = True,
    ) -> bool:
        """Level 2: Automated promotion gate reviewing, approving, and releasing models.

        Compares candidate/challenger version against current champion in Model Registry.
        If approval gate criteria are met, promotes candidate to '@champion' / 'Production'.

        Args:
            registered_model_name: Model name in registry
            challenger_version: Version of candidate model to evaluate
            metric_name: Performance metric to gate on (e.g., 'rmse', 'accuracy')
            metric_threshold: Required improvement threshold over baseline
            lower_is_better: True for loss/error metrics (e.g., RMSE), False for accuracy/R2

        Returns:
            bool: True if promoted, False if rejected
        """
        logger.info(
            f"Evaluating promotion gate for {registered_model_name} version {challenger_version} on {metric_name}"
        )

        try:
            # Check current champion version
            champion_version_info = self.client.get_model_version_by_alias(
                name=registered_model_name,
                alias="champion",
            )
            logger.info(f"Current champion version: {champion_version_info.version}")
        except Exception:
            logger.info("No current champion found. Automatically promoting initial model version.")
            self.promote_to_champion(registered_model_name, challenger_version)
            return True

        # Tag candidate as challenger stage
        self.client.set_registered_model_alias(
            name=registered_model_name,
            alias="challenger",
            version=challenger_version,
        )

        # Gate check: retrieve metrics from challenger and champion runs
        challenger_version_info = self.client.get_model_version(
            name=registered_model_name,
            version=challenger_version,
        )
        challenger_run = self.client.get_run(challenger_version_info.run_id)
        champion_run = self.client.get_run(champion_version_info.run_id)

        challenger_metric = challenger_run.data.metrics.get(metric_name)
        champion_metric = champion_run.data.metrics.get(metric_name)

        if challenger_metric is None or champion_metric is None:
            logger.warning(
                f"Metric '{metric_name}' missing from runs. Challenger: {challenger_metric}, "
                f"Champion: {champion_metric}. Gating model promotion to fail safe."
            )
            return False

        if lower_is_better:
            promoted = challenger_metric < (champion_metric - metric_threshold)
        else:
            promoted = challenger_metric > (champion_metric + metric_threshold)

        logger.info(
            f"Evaluation result for {metric_name}: challenger={challenger_metric:.4f}, "
            f"champion={champion_metric:.4f}, threshold={metric_threshold}, promoted={promoted}"
        )

        if promoted:
            self.promote_to_champion(registered_model_name, challenger_version)
            self.client.set_model_version_tag(
                name=registered_model_name,
                version=challenger_version,
                key="approval_gate",
                value="approved",
            )
            return True
        return False

    def promote_to_champion(self, registered_model_name: str, version: str) -> None:
        """Promotes model version to '@champion' alias and 'Production' governance stage."""
        logger.info(f"Promoting {registered_model_name} v{version} to @champion alias (Production)")
        self.client.set_registered_model_alias(
            name=registered_model_name,
            alias="champion",
            version=str(version),
        )
        try:
            self.client.transition_model_version_stage(
                name=registered_model_name,
                version=str(version),
                stage="Production",
                archive_existing_versions=True,
            )
        except Exception:
            # Unity Catalog models use aliases rather than legacy stages
            pass

    def rollback_model(self, registered_model_name: str, fallback_version: str) -> None:
        """Level 2 Rollback Capability: Instantly rolls back champion alias to a verified previous version."""
        logger.warning(
            f"ROLLBACK TRIGGERED: Reverting champion alias for {registered_model_name} to version {fallback_version}"
        )
        self.client.set_registered_model_alias(
            name=registered_model_name,
            alias="champion",
            version=str(fallback_version),
        )
        self.client.set_model_version_tag(
            name=registered_model_name,
            version=str(fallback_version),
            key="governance_action",
            value="rollback_restored",
        )
