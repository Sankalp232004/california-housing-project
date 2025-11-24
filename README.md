## California Housing Risk Radar

This repository showcases a complete research-to-production workflow for assessing California housing valuations—framed for economics and finance roles where evidence-based investment or policy recommendations are key.

### Why it matters for hiring managers
- **Macro + micro view**: marries census microdata with domain features (affordability ratios, coastal exposure) to surface tract-level risk signals.
- **Reproducible analytics**: deterministic pipeline (`scripts/run_pipeline.py`) plus unit tests demonstrate engineering rigor and auditability.
- **Commercial insight**: feature importance highlights the levers (income, household density, coastal premium) that investors and policy analysts monitor when sizing supply-demand imbalances.

---

### Headline results (test set)

| Model | R² | RMSE (USD) | MAE (USD) | MAPE |
| --- | --- | --- | --- | --- |
| Elastic Net (baseline) | 0.64 | 69,156 | 49,633 | 28.8% |
| **HistGradientBoost (selected)** | **0.84** | **45,897** | **30,175** | **17.0%** |

Artifacts live in `reports/metrics.json` and `reports/figures/hist_gradient_boost_feature_importance.png`.

![Top price drivers](reports/figures/hist_gradient_boost_feature_importance.png)

Key economic takeaways:
- Median income remains the dominant price signal even after controlling for density and age effects.
- Household crowding (`population_per_household`) and supply constraints (`rooms_per_household`) explain variance inland, highlighting affordability stress pockets.
- Being within an hour of the ocean maintains a persistent premium despite 1990-era data, reinforcing the scarcity narrative for coastal permits.

---

### Repository tour

```
.
├── config/                 # YAML-configured experiment settings
├── docs/                   # Portfolio narratives & planning notes
├── notebook/               # Exploratory analysis (Jupyter)
├── reports/                # Auto-generated metrics + figures
├── scripts/                # CLI entry points (train pipeline)
├── src/housing_prices/     # Modular package: data, features, modeling
├── tests/                  # Regression + feature-engineering tests
├── Makefile                # One-line install / test / train commands
└── requirements.txt        # Runtime dependencies
```

Core modules:
- `config.py`: loads human-readable YAML for experiment parameters.
- `data.py` & `features.py`: handle ingestion, cleaning, and engineered ratios relevant to affordability analysis.
- `modeling.py`: compares regularized linear models vs. gradient boosting; stores metrics for resume-ready storytelling.
- `visualization.py`: automatically exports the most material drivers for slide decks or conversations.

---

### Quick start (macOS / Linux / WSL)

```bash
git clone https://github.com/Sankalp232004/california-housing-project.git
cd california-housing-project
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e '.[dev]'

# Train models & regenerate reports
python scripts/run_pipeline.py

# Run unit tests
pytest

# Optional: reproduce visuals in the notebook
code notebook/housing_analysis.ipynb
```

Prefer Make targets? `make install`, `make train`, `make test` are available.

---

### Methodology snapshot
1. **Feature engineering** – builds affordability ratios (`rooms_per_household`, `population_per_household`, `bedrooms_per_room`) and macro-style indicators (`income_to_age_ratio`, coastal dummy) to mimic due-diligence heuristics.
2. **Preprocessing** – ColumnTransformer with median imputation + scaling for numerics, frequency encoding for categories.
3. **Model comparison** – Elastic Net for interpretability and HistGradientBoost for non-linear uplift; selection based on lowest RMSE.
4. **Reporting** – writes JSON + PNG assets for portfolio decks and recruiters.

---

### Extending the work
- Swap `config/project.yaml` parameters (test split, hyper-parameters) to simulate different macro scenarios.
- Drop in fresh census/ACS feeds at `data/housing.csv`—the package infers schema automatically.
- Use the exported metrics and figure in presentations or add more notebooks for localized case studies.

For a concise storytelling aid, see `docs/portfolio_brief.md`. Implementation notes live in `docs/PROJECT_PLAN.md`.
