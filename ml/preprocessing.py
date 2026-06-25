"""Preprocessing pipeline construction.

Builds a ``ColumnTransformer`` that imputes, scales numeric features, and
one-hot encodes categoricals. Always imputes (median / most-frequent) so the
pipeline never crashes on missing values, and handles the case where one branch
(all-numeric or all-categorical data) is empty.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler


def build_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
    scaler: str = "standard",
) -> ColumnTransformer:
    """Build a ColumnTransformer for the given feature columns.

    Args:
        numeric_cols: numeric feature column names.
        categorical_cols: categorical feature column names.
        scaler: "standard" (StandardScaler) or "robust" (RobustScaler). The
            robust scaler is used during the skew-fix improvement round.
    """
    scaler_obj = RobustScaler() if scaler == "robust" else StandardScaler()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", scaler_obj),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", max_categories=20),
            ),
        ]
    )

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipeline, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipeline, categorical_cols))

    # remainder="drop" guards against the target leaking back into X.
    return ColumnTransformer(transformers=transformers, remainder="drop")
