"""Feature engineering logic for the housing portfolio."""

from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL_FEATURES = ["ocean_proximity"]


def engineer_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create ratio-based features with basic safeguards."""

    engineered = df.copy()
    households = engineered["households"].replace(0, np.nan)
    rooms = engineered["total_rooms"].replace(0, np.nan)

    engineered["rooms_per_household"] = engineered["total_rooms"] / households
    engineered["bedrooms_per_room"] = engineered["total_bedrooms"] / rooms
    engineered["population_per_household"] = engineered["population"] / households
    engineered["income_to_age_ratio"] = engineered["median_income"] / (
        engineered["housing_median_age"] + 1
    )
    engineered["coastal_flag"] = (
        engineered["ocean_proximity"].isin(["<1H OCEAN", "NEAR OCEAN"])
    ).astype(int)

    engineered.replace([np.inf, -np.inf], np.nan, inplace=True)
    return engineered


def _categorical_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in df.columns if df[col].dtype == "object"]


def _numeric_columns(df: pd.DataFrame, *, exclude: Iterable[str] = ()) -> List[str]:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    return [col for col in numeric_cols if col not in exclude]


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """Create a reusable preprocessing pipeline for models."""

    categorical_cols = _categorical_columns(df)
    numeric_cols = _numeric_columns(df, exclude=categorical_cols)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_cols),
            ("categorical", categorical_pipeline, categorical_cols),
        ]
    )


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """Return the expanded feature names after preprocessing."""

    feature_names: List[str] = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == "remainder":
            continue
        if name == "numeric":
            feature_names.extend(cols)
        elif name == "categorical":
            encoder = transformer.named_steps["encoder"]
            encoded_cols = encoder.get_feature_names_out(cols)
            feature_names.extend(encoded_cols.tolist())
    return feature_names
