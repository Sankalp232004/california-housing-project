"""Command-line entry point for the modeling pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from housing_prices.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Train and evaluate California housing models."
	)
	parser.add_argument(
		"--config",
		"-c",
		default=Path("config/project.yaml"),
		type=Path,
		help="Path to a YAML config file.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	result = run_pipeline(args.config)
	print("✅ Pipeline finished")
	print(f"Best model: {result.best_model}")
	print(f"Metrics written to: {result.metrics_path}")
	print(f"Feature importance saved to: {result.feature_figure_path}")


if __name__ == "__main__":
	main()
