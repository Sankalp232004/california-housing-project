"""Model training and evaluation helpers."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.pipeline import Pipeline

from .config import ProjectConfig


def build_model_registry(
    preprocessor: Pipeline, config: ProjectConfig
) -> Dict[str, Pipeline]:
    """Return a dictionary of candidate models keyed by readable names."""

    elastic_net = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                ElasticNet(
                    alpha=config.alpha,
                    l1_ratio=config.l1_ratio,
                    max_iter=10_000,
                    random_state=config.random_state,
                ),
            ),
        ]
    )

    gradient_boosting = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                HistGradientBoostingRegressor(
                    learning_rate=config.gradient_boosting.learning_rate,
                    max_depth=config.gradient_boosting.max_depth,
                    max_iter=config.gradient_boosting.n_estimators,
                    random_state=config.random_state,
                ),
            ),
        ]
    )

    return {
        "elastic_net": elastic_net,
        "hist_gradient_boost": gradient_boosting,
    }


def fit_models(
    models: Dict[str, Pipeline], X_train: pd.DataFrame, y_train: pd.Series
) -> Dict[str, Pipeline]:
    """Fit every model in place and return the same mapping."""

    for name, model in models.items():
        model.fit(X_train, y_train)
    return models


def _regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = root_mean_squared_error(y_true, y_pred)
    return {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": rmse,
        "mape": float(np.mean(np.abs((y_true - y_pred) / y_true))) * 100,
    }


def evaluate_models(
    models: Dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Dict[str, float]]:
    """Compute evaluation metrics for each trained model."""

    metrics: Dict[str, Dict[str, float]] = {}
    for name, model in models.items():
        predictions = model.predict(X_test)
        metrics[name] = _regression_metrics(y_test, predictions)
    return metrics


def select_best_model(
    metrics: Dict[str, Dict[str, float]]
) -> Tuple[str, Dict[str, float]]:
    """Return the name/metrics of the model with the lowest RMSE."""

    if not metrics:
        raise ValueError("No metrics provided for selection")
    best_name = min(metrics, key=lambda key: metrics[key]["rmse"])
    return best_name, metrics[best_name]
