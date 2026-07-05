"""
Feature Engineering Module
──────────────────────────────────────────────────────────────────────────────
Implements the two-phase feature strategy:
  Phase 1: Broad transformation (raw → 25K-30K transformed features)
  Phase 2: Model-specific feature selection (30K → 1K-3K per model)

For this demo, transforms ~50 raw features → ~200 engineered features,
then selects top 30 per model.

In production (Databricks), this would use:
  - PySpark ML Pipelines for distributed transforms
  - Databricks Feature Store (Unity Catalog) for registration
  - Point-in-time lookups for training data assembly
"""

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
    f_regression,
    mutual_info_classif,
)
from sklearn.preprocessing import (
    KBinsDiscretizer,
    LabelEncoder,
    PolynomialFeatures,
    StandardScaler,
)

from src.utils import (
    get_model_type_category,
    load_parquet,
    logger,
    save_parquet,
    timer,
)


class FeatureEngineeringPipeline:
    """
    Encapsulates feature transformation and selection logic.

    This class mirrors the PySpark ML Pipeline pattern: fit on training data,
    then transform both training and inference data identically to prevent
    training-serving skew.
    """

    def __init__(self, config: dict):
        self.config = config
        self.feature_config = config["features"]
        self.scalers: dict[str, StandardScaler] = {}
        self.encoders: dict[str, LabelEncoder] = {}
        self.binner: Optional[KBinsDiscretizer] = None
        self.numeric_columns: list[str] = []
        self.categorical_columns: list[str] = []
        self.temporal_columns: list[str] = []
        self.text_columns: list[str] = []
        self.fitted = False

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit transformers on training data and return transformed features."""
        with timer("Feature Engineering — Fit & Transform"):
            self._identify_columns(df)
            result = self._apply_transforms(df, fit=True)
            self.fitted = True
            logger.info(
                f"Feature engineering complete: "
                f"{df.shape[1]} raw → {result.shape[1]} transformed features"
            )
            return result

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted transformers (inference)."""
        if not self.fitted:
            raise RuntimeError(
                "Pipeline not fitted. Call fit_transform() first."
            )
        with timer("Feature Engineering — Transform (Inference)"):
            result = self._apply_transforms(df, fit=False)
            logger.info(
                f"Inference transform: "
                f"{df.shape[1]} raw → {result.shape[1]} features"
            )
            return result

    def select_features(
        self,
        df: pd.DataFrame,
        target: Optional[pd.Series],
        model_type: str,
        max_features: int = 30,
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Select top features for a specific model.

        Args:
            df: Transformed feature DataFrame
            target: Target variable (None for clustering)
            model_type: Model type string from config
            max_features: Maximum features to select

        Returns:
            tuple: (selected_df, selected_feature_names)
        """
        category = get_model_type_category(model_type)

        # For clustering, use variance-based selection
        if category == "clusterer" or target is None:
            variances = df.var().sort_values(ascending=False)
            selected_cols = variances.head(max_features).index.tolist()
            logger.info(
                f"Feature selection (variance): {df.shape[1]} → {len(selected_cols)} features"
            )
            return df[selected_cols], selected_cols

        # Remove rows with NaN in target for selection
        valid_mask = ~target.isna()
        df_valid = df[valid_mask]
        target_valid = target[valid_mask]

        # Handle remaining NaNs in features
        df_valid = df_valid.fillna(0)

        # Limit features to available count
        k = min(max_features, df_valid.shape[1])

        if category == "classifier":
            selector = SelectKBest(score_func=f_classif, k=k)
        else:
            selector = SelectKBest(score_func=f_regression, k=k)

        try:
            selector.fit(df_valid, target_valid)
            mask = selector.get_support()
            selected_cols = df.columns[mask].tolist()
        except Exception as e:
            logger.warning(
                f"Feature selection failed ({e}), falling back to variance selection"
            )
            variances = df.var().sort_values(ascending=False)
            selected_cols = variances.head(max_features).index.tolist()

        logger.info(
            f"Feature selection ({category}): "
            f"{df.shape[1]} → {len(selected_cols)} features"
        )
        return df[selected_cols], selected_cols

    # ── Internal Methods ─────────────────────────────────────────────────

    def _identify_columns(self, df: pd.DataFrame) -> None:
        """Categorize columns by type."""
        self.numeric_columns = [
            c for c in df.columns
            if c.startswith("num_") or c.startswith("text_len") or c.startswith("text_words")
        ]
        self.categorical_columns = [
            c for c in df.columns if c.startswith("cat_")
        ]
        self.temporal_columns = [
            c for c in df.columns if c.startswith("date_")
        ]
        self.text_columns = [
            c for c in df.columns
            if c.startswith("text_") and not c.startswith("text_len") and not c.startswith("text_words")
        ]

        logger.info(
            f"Column types: {len(self.numeric_columns)} numeric, "
            f"{len(self.categorical_columns)} categorical, "
            f"{len(self.temporal_columns)} temporal, "
            f"{len(self.text_columns)} text"
        )

    def _apply_transforms(
        self, df: pd.DataFrame, fit: bool
    ) -> pd.DataFrame:
        """Apply all feature transformations."""
        features = {}

        # Keep record_id for joining
        if "record_id" in df.columns:
            features["record_id"] = df["record_id"]

        # 1. Numeric transforms
        numeric_features = self._transform_numeric(df, fit)
        features.update(numeric_features)

        # 2. Categorical transforms
        categorical_features = self._transform_categorical(df, fit)
        features.update(categorical_features)

        # 3. Temporal transforms
        temporal_features = self._transform_temporal(df)
        features.update(temporal_features)

        result = pd.DataFrame(features)

        # Drop non-feature columns for the feature matrix
        feature_cols = [c for c in result.columns if c != "record_id"]
        return result[feature_cols]

    def _transform_numeric(
        self, df: pd.DataFrame, fit: bool
    ) -> dict[str, np.ndarray]:
        """Apply numeric transformations: scaling, binning, interactions."""
        features = {}

        for col in self.numeric_columns:
            if col not in df.columns:
                continue

            values = df[col].values.astype(float)
            values_filled = np.nan_to_num(values, nan=0.0)

            # Original (imputed)
            features[f"{col}_orig"] = values_filled

            # Null indicator
            features[f"{col}_is_null"] = np.isnan(values).astype(float)

            # Standard scaling
            if fit:
                scaler = StandardScaler()
                scaled = scaler.fit_transform(values_filled.reshape(-1, 1)).ravel()
                self.scalers[col] = scaler
            else:
                if col in self.scalers:
                    scaled = self.scalers[col].transform(
                        values_filled.reshape(-1, 1)
                    ).ravel()
                else:
                    scaled = values_filled
            features[f"{col}_scaled"] = scaled

            # Log transform (for positive values)
            features[f"{col}_log1p"] = np.log1p(np.abs(values_filled))

            # Squared term
            features[f"{col}_squared"] = values_filled ** 2

        # Pairwise interactions (top K numeric features by variance)
        top_k = self.config["features"]["transforms"]["numeric"].get(
            "interaction_top_k", 10
        )
        top_numeric = sorted(
            self.numeric_columns[:top_k],
            key=lambda c: np.nanvar(df[c]) if c in df.columns else 0,
            reverse=True,
        )[:min(top_k, 5)]  # Limit for demo

        for i in range(len(top_numeric)):
            for j in range(i + 1, len(top_numeric)):
                col_a = top_numeric[i]
                col_b = top_numeric[j]
                if col_a in df.columns and col_b in df.columns:
                    a = np.nan_to_num(df[col_a].values.astype(float))
                    b = np.nan_to_num(df[col_b].values.astype(float))
                    features[f"interact_{col_a}_{col_b}"] = a * b

        return features

    def _transform_categorical(
        self, df: pd.DataFrame, fit: bool
    ) -> dict[str, np.ndarray]:
        """Apply categorical transformations: label encoding + frequency encoding."""
        features = {}

        for col in self.categorical_columns:
            if col not in df.columns:
                continue

            values = df[col].astype(str)

            # Label encoding
            if fit:
                encoder = LabelEncoder()
                # Fit including an "unknown" category for unseen values
                unique_vals = list(values.unique()) + ["__unknown__"]
                encoder.fit(unique_vals)
                self.encoders[col] = encoder

            if col in self.encoders:
                encoder = self.encoders[col]
                # Map unseen categories to __unknown__
                known = set(encoder.classes_)
                values_safe = values.map(
                    lambda x: x if x in known else "__unknown__"
                )
                features[f"{col}_encoded"] = encoder.transform(values_safe)
            else:
                features[f"{col}_encoded"] = pd.Categorical(values).codes

            # Frequency encoding
            freq_map = values.value_counts(normalize=True).to_dict()
            features[f"{col}_freq"] = values.map(freq_map).values

        return features

    def _transform_temporal(
        self, df: pd.DataFrame
    ) -> dict[str, np.ndarray]:
        """Extract temporal features from date columns."""
        features = {}

        for col in self.temporal_columns:
            if col not in df.columns:
                continue

            dates = pd.to_datetime(df[col], errors="coerce")

            # Basic components
            features[f"{col}_year"] = dates.dt.year.fillna(0).values.astype(float)
            features[f"{col}_month"] = dates.dt.month.fillna(0).values.astype(float)
            features[f"{col}_dayofweek"] = dates.dt.dayofweek.fillna(0).values.astype(float)
            features[f"{col}_dayofyear"] = dates.dt.dayofyear.fillna(0).values.astype(float)

            # Cyclical encoding (prevent ordinality issues)
            month_vals = dates.dt.month.fillna(1).values.astype(float)
            features[f"{col}_month_sin"] = np.sin(2 * np.pi * month_vals / 12)
            features[f"{col}_month_cos"] = np.cos(2 * np.pi * month_vals / 12)

            dow_vals = dates.dt.dayofweek.fillna(0).values.astype(float)
            features[f"{col}_dow_sin"] = np.sin(2 * np.pi * dow_vals / 7)
            features[f"{col}_dow_cos"] = np.cos(2 * np.pi * dow_vals / 7)

            # Days since reference date
            ref_date = pd.Timestamp("2024-01-01")
            features[f"{col}_days_since_ref"] = (
                (dates - ref_date).dt.days.fillna(0).values.astype(float)
            )

            # Is weekend
            features[f"{col}_is_weekend"] = (
                dates.dt.dayofweek.isin([5, 6]).fillna(False).values.astype(float)
            )

        return features


def run_feature_engineering(
    train_df: pd.DataFrame, inference_df: pd.DataFrame, config: dict
) -> tuple[pd.DataFrame, pd.DataFrame, FeatureEngineeringPipeline]:
    """
    Execute the full feature engineering pipeline.

    Returns:
        tuple: (train_features, inference_features, fitted_pipeline)
    """
    pipeline = FeatureEngineeringPipeline(config)

    # Phase 1: Fit on training data and transform
    train_features = pipeline.fit_transform(train_df)

    # Phase 2: Transform inference data using fitted pipeline
    inference_features = pipeline.transform(inference_df)

    # Save transformed features (Gold layer)
    paths = config["data"]["paths"]
    save_parquet(
        train_features,
        f"{paths['feature_data']}train_features.parquet",
        "Training features",
    )
    save_parquet(
        inference_features,
        f"{paths['feature_data']}inference_features.parquet",
        "Inference features",
    )

    return train_features, inference_features, pipeline
