import os
import tempfile

import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import mlflow
from mlflow.models import infer_signature
import ray
import ray.train
from ray.train import Checkpoint
from common.utils import get_logger

logger = get_logger()


def get_torch_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    """Builds train/test DataLoaders from in-memory numpy arrays."""
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(test_ds, batch_size=batch_size),
    )


def log_ray_result_to_mlflow(result) -> None:
    metrics = result.metrics or {}
    numeric_metrics = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float, np.integer, np.floating))}
    if numeric_metrics:
        mlflow.log_metrics(numeric_metrics)


def set_pytorch_checkpoint(model: nn.Module, metrics) -> None:
    with tempfile.TemporaryDirectory(dir="/dbfs/tmp") as temp_checkpoint_dir:
        checkpoint = None

        if ray.train.get_context().get_world_rank() == 0:
            torch.save(
                model.module.state_dict(),  # NOTE: Unwrap the model.
                os.path.join(temp_checkpoint_dir, "model.pt"),
            )
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)

        ray.train.report(metrics, checkpoint=checkpoint)
