PYTHON := python3

.PHONY: install format lint test train report

install:
	pip install -r requirements.txt
	pip install -e '.[dev]'

format:
	black src tests

lint:
	ruff check src tests

test:
	pytest

train:
	$(PYTHON) scripts/run_pipeline.py train

report: train
	@echo "Artifacts stored under reports/"
