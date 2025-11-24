"""Project-level configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(slots=True)
class GradientBoostingConfig:
    """Hyper-parameters for the gradient boosting model."""

    learning_rate: float = 0.1
    max_depth: int = 5
    n_estimators: int = 300


@dataclass(slots=True)
class ProjectConfig:
    """Container for tunable project settings."""

    project_name: str
    raw_data_path: Path
    target_column: str
    test_size: float
    random_state: int
    alpha: float
    l1_ratio: float
    gradient_boosting: GradientBoostingConfig
    top_feature_count: int
    report_dir: Path

    @property
    def report_directory(self) -> Path:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        return self.report_dir

    @property
    def figures_directory(self) -> Path:
        figures_path = self.report_directory / "figures"
        figures_path.mkdir(parents=True, exist_ok=True)
        return figures_path

    @property
    def metrics_path(self) -> Path:
        return self.report_directory / "metrics.json"


def _coerce_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        return Path.cwd() / path
    return path


def _load_yaml(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_config(path: str | Path = "config/project.yaml") -> ProjectConfig:
    """Parse YAML configuration into a strongly typed dataclass."""

    config_path = _coerce_path(path)
    raw_cfg = _load_yaml(config_path)

    gb_cfg = raw_cfg.get("gradient_boosting", {})
    gradient_boosting = GradientBoostingConfig(**gb_cfg)

    return ProjectConfig(
        project_name=raw_cfg["project_name"],
        raw_data_path=_coerce_path(raw_cfg["raw_data_path"]),
        target_column=raw_cfg["target_column"],
        test_size=float(raw_cfg["test_size"]),
        random_state=int(raw_cfg["random_state"]),
        alpha=float(raw_cfg["alpha"]),
        l1_ratio=float(raw_cfg["l1_ratio"]),
        gradient_boosting=gradient_boosting,
        top_feature_count=int(raw_cfg.get("top_feature_count", 10)),
        report_dir=_coerce_path(raw_cfg.get("report_dir", "reports")),
    )
