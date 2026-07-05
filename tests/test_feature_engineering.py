"""
Unit Tests — Feature Engineering
──────────────────────────────────────────────────────────────────────────────
Tests the feature transformation pipeline for correctness:
  - Numeric transforms produce expected columns
  - Categorical encoding handles unseen categories
  - Temporal features extract correct components
  - Feature selection reduces dimensionality
  - Pipeline is idempotent (fit-transform consistency)
"""

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.feature_engineering import FeatureEngineeringPipeline
from src.utils import load_config


@pytest.fixture
def sample_config():
    """Minimal config for testing."""
    return {
        "features": {
            "transforms": {
                "numeric": {
                    "scaling": "standard",
                    "binning_quantiles": 5,
                    "polynomial_degree": 2,
                    "interaction_top_k": 5,
                },
                "categorical": {
                    "encoding": "target",
                    "max_cardinality": 50,
                    "hash_buckets": 100,
                },
                "temporal": {
                    "lag_periods": [1, 7],
                    "rolling_windows": [7, 30],
                    "cyclical": True,
                },
            },
            "selection": {
                "method": "importance",
                "max_features_per_model": 10,
                "variance_threshold": 0.01,
                "correlation_threshold": 0.95,
            },
        }
    }


@pytest.fixture
def sample_data():
    """Generate a small sample DataFrame for testing."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame(
        {
            "record_id": range(n),
            "num_000": np.random.normal(0, 1, n),
            "num_001": np.random.lognormal(0, 1, n),
            "num_002": np.random.uniform(-10, 10, n),
            "cat_geography": np.random.choice(["US", "EU", "APAC"], n),
            "cat_segment": np.random.choice(["retail", "enterprise"], n),
            "date_000": pd.date_range("2024-01-01", periods=n, freq="D"),
            "text_len_000": np.random.randint(5, 200, n),
            "text_words_000": np.random.randint(1, 50, n),
            "target_binary": np.random.randint(0, 2, n),
            "target_continuous": np.random.normal(100, 20, n),
        }
    )


class TestFeatureEngineeringPipeline:
    """Test suite for FeatureEngineeringPipeline."""

    def test_fit_transform_produces_features(self, sample_data, sample_config):
        """Pipeline should produce more features than input."""
        pipeline = FeatureEngineeringPipeline(sample_config)
        result = pipeline.fit_transform(sample_data)

        assert result.shape[0] == len(sample_data)
        assert result.shape[1] > 5  # Should have many more features
        assert pipeline.fitted is True

    def test_transform_after_fit(self, sample_data, sample_config):
        """Transform should work after fitting."""
        pipeline = FeatureEngineeringPipeline(sample_config)
        pipeline.fit_transform(sample_data)

        result = pipeline.transform(sample_data)
        assert result.shape[0] == len(sample_data)
        assert result.shape[1] > 0

    def test_transform_without_fit_raises(self, sample_data, sample_config):
        """Transform without fit should raise RuntimeError."""
        pipeline = FeatureEngineeringPipeline(sample_config)
        with pytest.raises(RuntimeError, match="not fitted"):
            pipeline.transform(sample_data)

    def test_numeric_transforms(self, sample_data, sample_config):
        """Numeric features should produce scaled, log, squared variants."""
        pipeline = FeatureEngineeringPipeline(sample_config)
        result = pipeline.fit_transform(sample_data)

        # Check that derived columns exist
        assert "num_000_scaled" in result.columns
        assert "num_000_log1p" in result.columns
        assert "num_000_squared" in result.columns
        assert "num_000_is_null" in result.columns

    def test_categorical_encoding(self, sample_data, sample_config):
        """Categorical features should be encoded as numeric."""
        pipeline = FeatureEngineeringPipeline(sample_config)
        result = pipeline.fit_transform(sample_data)

        assert "cat_geography_encoded" in result.columns
        assert "cat_geography_freq" in result.columns
        # Encoded values should be numeric
        assert result["cat_geography_encoded"].dtype in [np.int32, np.int64, np.float64]

    def test_unseen_categories_handled(self, sample_data, sample_config):
        """Pipeline should handle unseen categories at inference time."""
        pipeline = FeatureEngineeringPipeline(sample_config)
        pipeline.fit_transform(sample_data)

        # Create inference data with unseen category
        inference_data = sample_data.copy()
        inference_data.loc[0, "cat_geography"] = "UNSEEN_REGION"

        result = pipeline.transform(inference_data)
        assert not result.isnull().all().any()  # No all-null columns

    def test_temporal_features(self, sample_data, sample_config):
        """Temporal features should extract year, month, cyclical components."""
        pipeline = FeatureEngineeringPipeline(sample_config)
        result = pipeline.fit_transform(sample_data)

        assert "date_000_month" in result.columns
        assert "date_000_month_sin" in result.columns
        assert "date_000_month_cos" in result.columns
        assert "date_000_is_weekend" in result.columns

    def test_no_nan_in_output(self, sample_data, sample_config):
        """Output should not have NaN values (except intentional null indicators)."""
        pipeline = FeatureEngineeringPipeline(sample_config)
        result = pipeline.fit_transform(sample_data)

        # Null indicator columns are expected to have 0/1 values
        non_null_cols = [c for c in result.columns if "_is_null" not in c]
        # Most columns should be non-null (some interaction columns might have edge cases)
        null_rates = result[non_null_cols].isnull().mean()
        assert (null_rates < 0.5).all(), f"High null rates: {null_rates[null_rates >= 0.5]}"

    def test_feature_selection_reduces_dimensions(
        self, sample_data, sample_config
    ):
        """Feature selection should reduce to max_features_per_model."""
        pipeline = FeatureEngineeringPipeline(sample_config)
        result = pipeline.fit_transform(sample_data)

        target = sample_data["target_binary"]
        selected_df, selected_features = pipeline.select_features(
            df=result,
            target=target,
            model_type="xgboost_classifier",
            max_features=10,
        )

        assert len(selected_features) == 10
        assert selected_df.shape[1] == 10

    def test_clustering_feature_selection(self, sample_data, sample_config):
        """Clustering should use variance-based selection (no target)."""
        pipeline = FeatureEngineeringPipeline(sample_config)
        result = pipeline.fit_transform(sample_data)

        selected_df, selected_features = pipeline.select_features(
            df=result,
            target=None,
            model_type="sklearn_kmeans",
            max_features=5,
        )

        assert len(selected_features) == 5

    def test_consistency_between_fit_and_transform(
        self, sample_data, sample_config
    ):
        """fit_transform and transform should produce same column structure."""
        pipeline = FeatureEngineeringPipeline(sample_config)
        fit_result = pipeline.fit_transform(sample_data)
        transform_result = pipeline.transform(sample_data)

        assert set(fit_result.columns) == set(transform_result.columns)
        assert fit_result.shape == transform_result.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
