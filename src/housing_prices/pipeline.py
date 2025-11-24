"""High-level orchestration of the modeling workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from .config import ProjectConfig, load_config
from .data import load_raw_data, make_train_test_split, split_features_targets
from .features import (
    build_preprocessor,
    engineer_domain_features,
    get_feature_names,
)
from .modeling import (
    build_model_registry,
    evaluate_models,
    fit_models,
    select_best_model,
)
from .visualization import extract_feature_importances, plot_feature_importance


@dataclass(slots=True)
class PipelineResult:
    metrics: Dict[str, Dict[str, float]]
    best_model: str
    feature_figure_path: Path
    metrics_path: Path


class PipelineRunner:
    """End-to-end pipeline wrapper for the housing project."""

    def __init__(self, config: Optional[ProjectConfig] = None) -> None:
        self.config = config or load_config()

    def run(self) -> PipelineResult:
        raw_df = load_raw_data(self.config)
        engineered_df = engineer_domain_features(raw_df)
        X, y = split_features_targets(engineered_df, self.config.target_column)

        X_train, X_test, y_train, y_test = make_train_test_split(
            X, y, config=self.config
        )

        preprocessor = build_preprocessor(X_train)
        models = build_model_registry(preprocessor, self.config)
        fitted_models = fit_models(models, X_train, y_train)
        metrics = evaluate_models(fitted_models, X_test, y_test)
        best_model_name, _ = select_best_model(metrics)

        fitted_preprocessor = fitted_models[best_model_name].named_steps["preprocessor"]
        feature_names = get_feature_names(fitted_preprocessor)
        importances = extract_feature_importances(
            fitted_models[best_model_name],
            feature_names,
            X_sample=X_test,
            y_sample=y_test,
        )
        figure_path = plot_feature_importance(
            importances,
            top_n=self.config.top_feature_count,
            output_path=self.config.figures_directory
            / f"{best_model_name}_feature_importance.png",
        )

        self._write_metrics(metrics, best_model_name)
        return PipelineResult(
            metrics=metrics,
            best_model=best_model_name,
            feature_figure_path=figure_path,
            metrics_path=self.config.metrics_path,
        )

    def _write_metrics(self, metrics: Dict[str, Dict[str, float]], best_model: str) -> None:
        payload = {"best_model": best_model, "metrics": metrics}
        self.config.report_directory.mkdir(parents=True, exist_ok=True)
        with self.config.metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


def run_pipeline(config_path: str | Path = "config/project.yaml") -> PipelineResult:
    """Convenience function for scripts."""

    config = load_config(config_path)
    runner = PipelineRunner(config)
    return runner.run()
