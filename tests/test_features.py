"""Unit tests for feature engineering utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from housing_prices.features import build_preprocessor, engineer_domain_features


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "longitude": [-120.0, -121.5],
            "latitude": [36.5, 34.2],
            "housing_median_age": [15, 30],
            "total_rooms": [1000, 800],
            "total_bedrooms": [200, 160],
            "population": [500, 400],
            "households": [250, 200],
            "median_income": [4.2, 6.1],
            "median_house_value": [200000, 300000],
            "ocean_proximity": ["INLAND", "NEAR OCEAN"],
        }
    )


def test_engineer_domain_features_creates_ratio_columns() -> None:
    df = engineer_domain_features(_sample_frame())
    expected_columns = {
        "rooms_per_household",
        "bedrooms_per_room",
        "population_per_household",
        "income_to_age_ratio",
        "coastal_flag",
    }
    assert expected_columns.issubset(df.columns)
    assert np.isfinite(df["rooms_per_household"]).all()
    assert (df["coastal_flag"].isin([0, 1])).all()


def test_build_preprocessor_handles_numeric_and_categorical() -> None:
    df = engineer_domain_features(_sample_frame())
    X = df.drop(columns=["median_house_value"])
    preprocessor = build_preprocessor(X)
    transformer_names = {name for name, *_ in preprocessor.transformers}
    assert {"numeric", "categorical"} <= transformer_names