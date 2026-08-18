import argparse
from pathlib import Path
from typing import Dict, List

import mlflow
import numpy as np
import ray
import ray.data
import ray.train
import torch
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.train import CheckpointConfig, FailureConfig, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer
from ray.util.spark import setup_ray_cluster, shutdown_ray_cluster
from torch import nn
from tqdm import tqdm

from common.mlflow_helper import (
    set_champion_alias_on_logged_model,
    start_mlflow_experiment_and_run,
)
from common.ray_helper import (
    log_ray_result_to_mlflow,
    set_pytorch_checkpoint,
)
from common.utils import get_logger

logger = get_logger()


# Model Definition
class AnimeScoreNet(nn.Module):
    """Tiny MLP for predicting anime Score from one-hot genre features."""

    def __init__(self, input_dim: int, hidden: int = 64, y_mean: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        # Start with the predict-the-mean baseline so the model converges
        # quickly without needing target standardization.
        nn.init.constant_(self.net[-1].bias, y_mean)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _batch_to_xy(batch: Dict[str, torch.Tensor], feature_cols: List[str], target_col: str) -> tuple:
    """Stacks per-column tensors from a Ray Data batch into an (X, y) pair."""
    X = torch.stack([batch[c] for c in feature_cols], dim=1)
    y = batch[target_col]
    return X, y


def train_func_per_worker(config: Dict):
    lr = config["lr"]
    epochs = config["epochs"]
    batch_size = config["batch_size_per_worker"]
    y_mean = config["y_mean"]
    feature_cols: List[str] = config["feature_cols"]
    target_col: str = config["target_col"]

    train_shard = ray.train.get_dataset_shard("train")
    test_shard = ray.train.get_dataset_shard("test")

    model = AnimeScoreNet(input_dim=len(feature_cols), y_mean=y_mean)
    model = ray.train.torch.prepare_model(model)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for batch in tqdm(
            train_shard.iter_torch_batches(batch_size=batch_size, dtypes=torch.float32),
            desc=f"Train Epoch {epoch}",
        ):
            X, y = _batch_to_xy(batch, feature_cols, target_col)
            pred = model(X)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        sq_err, n_total = 0.0, 0
        with torch.no_grad():
            for batch in tqdm(
                test_shard.iter_torch_batches(batch_size=batch_size, dtypes=torch.float32),
                desc=f"Test Epoch {epoch}",
            ):
                X, y = _batch_to_xy(batch, feature_cols, target_col)
                pred = model(X)
                sq_err += ((pred - y) ** 2).sum().item()
                n_total += y.shape[0]

        rmse = float(np.sqrt(sq_err / max(n_total, 1)))
        logger.info(f"epoch {epoch+1}/{epochs}  test_rmse={rmse:.4f}")
        set_pytorch_checkpoint(model, {"test_rmse": rmse, "epoch": epoch + 1})


def train_model(
    train_ds: ray.data.Dataset,
    test_ds: ray.data.Dataset,
    feature_cols: List[str],
    target_col: str,
    y_mean: float,
    experiment_path: str,
    num_workers: int = 2,
    use_gpu: bool = True,
    epochs: int = 30,
    learning_rate: float = 1e-3,
    global_batch_size: int = 256,
    storage_path: str = "/dbfs/tmp/ray_results/anime_score_predictor",
):
    train_config = {
        "lr": learning_rate,
        "epochs": epochs,
        "y_mean": y_mean,
        "batch_size_per_worker": global_batch_size // num_workers,
        "feature_cols": feature_cols,
        "target_col": target_col,
    }

    # Configure computation resources
    scaling_config = ScalingConfig(num_workers=num_workers, use_gpu=use_gpu)

    # Initialize a Ray TorchTrainer. Datasets are sharded automatically across
    # workers via `get_dataset_shard` inside the training function.
    trainer = TorchTrainer(
        train_loop_per_worker=train_func_per_worker,
        train_loop_config=train_config,
        scaling_config=scaling_config,
        datasets={"train": train_ds, "test": test_ds},
        run_config=RunConfig(
            failure_config=FailureConfig(max_failures=1),
            callbacks=[
                MLflowLoggerCallback(
                    experiment_name=experiment_path,
                    save_artifact=True,
                )
            ],
            checkpoint_config=CheckpointConfig(
                num_to_keep=1, checkpoint_score_attribute="test_rmse", checkpoint_score_order="min"
            ),
            storage_path=storage_path,
        ),
    )

    result = trainer.fit()
    logger.info(f"Training result: {result}")
    logger.info(f"View detailed results here: {result.path}")

    return result


def _parse_args() -> argparse.Namespace:
    """Parses and returns CLI arguments for the Ray distributed training pipeline.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Run the Ray distributed model training pipeline.")
    parser.add_argument("--catalog", required=True, help="Unity Catalog name.")
    parser.add_argument("--schema", required=True, help="Schema (database) name.")
    parser.add_argument("--feature_store_table_name", required=True, help="Feature store table name.")
    parser.add_argument("--model_name", required=True, help="Registered model name.")
    parser.add_argument("--experiment_path", required=True, help="MLflow experiment path.")
    parser.add_argument("--target_col", required=False, default="Score", help="Target column name.")
    parser.add_argument("--id_col", required=False, default="Name", help="ID column name.")
    parser.add_argument("--model_alias", required=False, default="champion", help="Alias for the registered model.")
    parser.add_argument("--num_workers", required=False, type=int, default=2, help="Number of Ray training workers.")
    parser.add_argument(
        "--use_gpu",
        required=False,
        default="true",
        choices=["true", "false"],
        help="Whether to use GPUs for training.",
    )
    parser.add_argument(
        "--num_cpus_worker_node",
        required=False,
        type=int,
        default=3,
        help="CPUs per Ray worker node. Leave at least 1 CPU per Spark worker for Spark itself so ray.data.from_spark can still schedule Spark tasks.",
    )
    parser.add_argument("--num_gpus_worker_node", required=False, type=int, default=1, help="GPUs per Ray worker node.")
    parser.add_argument("--min_worker_nodes", required=False, type=int, default=2, help="Min Ray worker nodes.")
    parser.add_argument("--max_worker_nodes", required=False, type=int, default=2, help="Max Ray worker nodes.")
    parser.add_argument(
        "--memory_worker_node",
        required=False,
        type=int,
        default=6 * 1024 * 1024 * 1024,
        help=(
            "Heap memory (bytes) reserved for each Ray worker. Keep "
            "spark_executor_memory + (heap + object_store_memory) < 0.8 * worker_physical_memory."
        ),
    )
    parser.add_argument(
        "--object_store_memory_worker_node",
        required=False,
        type=int,
        default=2 * 1024 * 1024 * 1024,
        help="Object-store memory (bytes) reserved for each Ray worker.",
    )
    parser.add_argument(
        "--num_cpus_head_node",
        required=False,
        type=int,
        default=2,
        help="CPUs Ray reserves on the driver/head node.",
    )
    parser.add_argument(
        "--memory_head_node",
        required=False,
        type=int,
        default=4 * 1024 * 1024 * 1024,
        help="Heap memory (bytes) for the Ray head node.",
    )
    parser.add_argument(
        "--object_store_memory_head_node",
        required=False,
        type=int,
        default=2 * 1024 * 1024 * 1024,
        help="Object-store memory (bytes) for the Ray head node.",
    )
    parser.add_argument("--epochs", required=False, type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--learning_rate", required=False, type=float, default=1e-3, help="Optimizer learning rate.")
    parser.add_argument(
        "--global_batch_size", required=False, type=int, default=256, help="Global batch size across all workers."
    )
    parser.add_argument(
        "--ray_storage_path",
        required=False,
        default="/dbfs/tmp/ray_results/anime_score_predictor",
        help="Storage path for Ray training results.",
    )
    parser.add_argument(
        "--ray_log_path",
        required=False,
        default="/dbfs/tmp/raylogs/scm-ray-cluster",
        help="Path to collect Ray cluster logs.",
    )
    parser.add_argument(
        "--test_size",
        required=False,
        type=float,
        default=0.2,
        help="Fraction of data to use for the test split (Spark randomSplit).",
    )
    parser.add_argument(
        "--random_seed",
        required=False,
        type=int,
        default=42,
        help="Seed for Spark randomSplit.",
    )
    return parser.parse_args()


def prepare_spark_train_test_split(
    spark: SparkSession,
    catalog: str,
    schema: str,
    feature_store_table_name: str,
    target_col: str,
    id_col: str | None,
    test_size: float,
    random_seed: int,
) -> tuple:
    """Reads the feature table, casts to float32, drops nulls, and produces
    train/test Spark DataFrames along with the predict-the-mean baseline.

    All work stays in Spark — no driver-side ``toPandas`` or numpy collect.

    Returns:
        Tuple of ``(train_df, test_df, feature_cols, y_mean, baseline_rmse)``.
    """
    logger.info("Reading feature store table from Unity Catalog")
    df = spark.read.table(f"{catalog}.{schema}.{feature_store_table_name}")

    drop_cols = {target_col} | ({id_col} if id_col else set())
    feature_cols = [c for c in df.columns if c not in drop_cols]
    # Keep only the columns we need and ensure all are float32-compatible.
    df = df.select(*[F.col(c).cast("float").alias(c) for c in feature_cols + [target_col]])
    # Drop rows with NaN/null in target or features.
    df = df.dropna()

    logger.info("Splitting train/test with Spark randomSplit (distributed)")
    train_df, test_df = df.randomSplit([1.0 - test_size, test_size], seed=random_seed)

    # Compute target stats and predict-the-mean baseline RMSE via Spark aggregates.
    train_stats = train_df.agg(
        F.mean(target_col).alias("mean"),
        F.stddev(target_col).alias("std"),
        F.min(target_col).alias("min"),
        F.max(target_col).alias("max"),
    ).first()
    y_mean = float(train_stats["mean"])
    logger.info(
        f"target stats: mean={y_mean:.3f} std={float(train_stats['std']):.3f} "
        f"min={float(train_stats['min']):.3f} max={float(train_stats['max']):.3f}"
    )
    baseline_rmse = float(
        test_df.select(F.sqrt(F.mean((F.col(target_col) - F.lit(y_mean)) ** 2)).alias("rmse")).first()["rmse"]
    )
    logger.info(f"predict-the-mean baseline test RMSE = {baseline_rmse:.4f}")

    return train_df, test_df, feature_cols, y_mean, baseline_rmse


def main():
    """Entry point for the Ray distributed training script."""
    logger.info("Starting Ray distributed model training pipeline.")
    args = _parse_args()

    use_gpu = args.use_gpu.lower() == "true"
    fully_qualified_model_name = f"{args.catalog}.{args.schema}.{args.model_name}"

    spark = SparkSession.getActiveSession()
    train_df, test_df, feature_cols, y_mean, baseline_rmse = prepare_spark_train_test_split(
        spark=spark,
        catalog=args.catalog,
        schema=args.schema,
        feature_store_table_name=args.feature_store_table_name,
        target_col=args.target_col,
        id_col=args.id_col,
        test_size=args.test_size,
        random_seed=args.random_seed,
    )

    logger.info("starting Ray cluster")
    setup_ray_cluster(
        num_gpus_worker_node=args.num_gpus_worker_node,
        min_worker_nodes=args.min_worker_nodes,
        max_worker_nodes=args.max_worker_nodes,
        num_cpus_worker_node=args.num_cpus_worker_node,
        memory_worker_node=args.memory_worker_node,
        object_store_memory_worker_node=args.object_store_memory_worker_node,
        num_cpus_head_node=args.num_cpus_head_node,
        memory_head_node=args.memory_head_node,
        object_store_memory_head_node=args.object_store_memory_head_node,
        collect_log_to_path=args.ray_log_path,
    )

    ray.init(address="auto", ignore_reinit_error=True)

    # Build Ray Datasets directly from the Spark DataFrames
    # round-trip and no driver-side collect.
    logger.info("Converting Spark DwataFrames to Ray Datasets via ray.data.from_spark")
    train_ds = ray.data.from_spark(train_df)
    test_ds = ray.data.from_spark(test_df)

    logger.info("Starting experiment")
    start_mlflow_experiment_and_run(experiment_path=args.experiment_path)
    mlflow.pytorch.autolog(log_models=False)

    try:
        logger.info("Training model")
        model_run = train_model(
            train_ds=train_ds,
            test_ds=test_ds,
            feature_cols=feature_cols,
            target_col=args.target_col,
            y_mean=y_mean,
            experiment_path=args.experiment_path,
            num_workers=args.num_workers,
            use_gpu=use_gpu,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            global_batch_size=args.global_batch_size,
            storage_path=args.ray_storage_path,
        )
        logger.info(f"Ray results path: {model_run.path}")

        log_ray_result_to_mlflow(model_run)
        mlflow.log_param("ray_results_path", model_run.path)
        mlflow.log_metric("baseline_rmse", baseline_rmse)

        # Final test RMSE comes from the last reported checkpoint metric — no
        # driver-side recomputation needed.
        final_rmse = float(model_run.metrics.get("test_rmse", float("nan")))
        logger.info(f"final test RMSE = {final_rmse:.4f}  (baseline = {baseline_rmse:.4f})")
        mlflow.log_metric("final_test_rmse", final_rmse)

        # Reconstruct model from the best checkpoint for registration.
        with model_run.checkpoint.as_directory() as checkpoint_dir:
            state_dict = torch.load(Path(checkpoint_dir) / "model.pt", map_location="cpu")
        input_dim = state_dict["net.0.weight"].shape[1]
        model = AnimeScoreNet(input_dim=input_dim, y_mean=y_mean)
        model.load_state_dict(state_dict)
        model.eval()

        # Take a tiny batch from the Ray test dataset purely to build the
        # MLflow model signature (not a full collect).
        sample_batch = test_ds.take_batch(batch_size=8)
        X_sample = np.stack([np.asarray(sample_batch[c], dtype=np.float32) for c in feature_cols], axis=1)
        with torch.no_grad():
            preds_sample = model(torch.from_numpy(X_sample)).cpu().numpy()
        signature = infer_signature(X_sample, preds_sample)

        mlflow.pytorch.log_model(
            pytorch_model=model,
            name="model",
            signature=signature,
            registered_model_name=fully_qualified_model_name,
        )

        set_champion_alias_on_logged_model(
            fully_qualified_model_name=fully_qualified_model_name,
            model_alias=args.model_alias,
        )

    finally:
        logger.info("Ending MLflow run.")
        if mlflow.active_run() is not None:
            mlflow.end_run()

        logger.info("Shutting down ray cluster")
        shutdown_ray_cluster()
