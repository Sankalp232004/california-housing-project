# Portfolio Brief: California Housing Risk Radar

## Elevator Pitch
A reproducible analytics stack that stress-tests California housing valuations using census microdata, investment-style feature engineering, and transparent model governance.

## Talking Points for Interviews
- **Economic intuition first**: Derived crowding, affordability, and coastal premium indicators mirror the metrics used by REIT analysts and public policy teams.
- **Deterministic workflow**: `scripts/run_pipeline.py` rebuilds the entire study (data prep ➜ modeling ➜ reporting) in under two minutes, producing shareable JSON + PNG artifacts.
- **Model accountability**: Baseline Elastic Net provides explainability, while Gradient Boosting supplies performance headroom; both are logged with reproducible seeds.
- **Actionable insights**: Highest residual risk sits where median income lags supply (high `population_per_household`), signaling where subsidies or targeted lending could bite.

## Suggested Demo Flow
1. Open `reports/figures/hist_gradient_boost_feature_importance.png` to discuss macro drivers.
2. Walk through `config/project.yaml` to show how scenario tweaks are versioned.
3. Run `python scripts/run_pipeline.py` live to highlight automation.
4. Reference `tests/test_features.py` to underscore quality habits.

## Next Experiments
- Layer American Community Survey updates to monitor affordability shifts post-2010.
- Attach postcode-level mortgage rate shocks to simulate portfolio stress testing.
- Wire the pipeline into a lightweight dashboard for recruiters who prefer interactive demos.
