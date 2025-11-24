"""Visualization helpers for reports."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import permutation_importance


def extract_feature_importances(
    model,
    feature_names: Iterable[str],
    *,
    X_sample=None,
    y_sample=None,
) -> Dict[str, float]:
    """Return absolute feature importances given a fitted pipeline."""

    estimator = model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importances = np.abs(estimator.coef_)
    elif X_sample is not None and y_sample is not None:
        result = permutation_importance(
            model,
            X_sample,
            y_sample,
            n_repeats=10,
            random_state=42,
            n_jobs=-1,
        )
        importances = result.importances_mean
    else:
        raise AttributeError(
            "Model does not expose feature importances or coefficients and no data was provided for permutation importance."
        )

    return dict(zip(feature_names, importances))


def plot_feature_importance(
    importances: Dict[str, float],
    *,
    top_n: int,
    output_path: Path,
) -> Path:
    """Persist a horizontal bar chart of the strongest features."""

    sorted_items = sorted(importances.items(), key=lambda item: item[1], reverse=True)[
        :top_n
    ]
    labels, values = zip(*sorted_items)

    plt.figure(figsize=(8, 4))
    bars = plt.barh(labels, values, color="#2a9d8f")
    plt.xlabel("Importance (arbitrary units)")
    plt.title("Top Drivers of California Housing Prices")
    plt.gca().invert_yaxis()

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_width() + max(values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path
