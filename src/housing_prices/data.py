"""Data ingestion helper functions."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import ProjectConfig


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def load_raw_data(config: ProjectConfig) -> pd.DataFrame:
    """Load the California housing dataset from disk."""

    dataset_path = _resolve_path(config.raw_data_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Expected dataset at {dataset_path} not found")
    return pd.read_csv(dataset_path)


def split_features_targets(
    df: pd.DataFrame, target_column: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split a dataframe into feature matrix and target vector."""

    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not present in dataframe")
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def make_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    config: ProjectConfig,
):
    """Return deterministic train/test splits for features and targets."""

    return train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
    )
