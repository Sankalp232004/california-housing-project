# California Housing Portfolio Refresh

## Objectives
- Present a polished, economics-focused narrative backed by rigorous data science practices.
- Provide reproducible pipelines covering ingestion, feature engineering, modeling, evaluation, and reporting.
- Highlight business relevance for investment, urban planning, and affordability policy discussions.

## Target Deliverables
1. **Modular codebase** (`src/housing_prices`) exposing data prep, modeling, and visualization utilities plus a Typer CLI for quick experimentation.
2. **Reproducible workflow** powered by a `Makefile`, pinned dependencies, and lightweight configuration via `yaml`.
3. **Reporting assets** such as generated metrics (`reports/metrics.json`) and publication-ready figures in `reports/figures/` for portfolio use.
4. **Professional documentation** (README refresh + `docs/portfolio_brief.md`) tailored to economic/finance audiences, emphasizing insights and ROI.
5. **Quality signals** via unit tests and lint-ready project layout.

## Planned Structure
```
.
├── docs/
│   ├── PROJECT_PLAN.md (this file)
│   └── portfolio_brief.md
├── notebook/
├── reports/
│   ├── figures/
│   └── metrics.json
├── scripts/
│   └── run_pipeline.py
├── src/housing_prices/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── features.py
│   ├── modeling.py
│   ├── pipeline.py
│   └── visualization.py
├── tests/
├── Makefile
├── pyproject.toml (optional) + requirements.txt
└── README.md (overhauled)
```

## Notes
- Keep raw data in `data/housing.csv` but expose path hooks for future data drops.
- Default model: regularized linear regression baseline + gradient boosting comparison to illustrate modeling depth.
- Reporting: store evaluation metrics + feature importance chart to reuse in applications.
