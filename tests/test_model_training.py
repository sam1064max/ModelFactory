"""
Unit Tests — Model Training
──────────────────────────────────────────────────────────────────────────────
Tests model creation, training, evaluation, and MLflow integration.
"""

import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import get_model_type_category


@pytest.fixture
def classification_data():
    """Generate synthetic classification data."""
    np.random.seed(42)
    n = 500
    X = pd.DataFrame({f"feature_{i}": np.random.randn(n) for i in range(20)})
    y = pd.Series((X["feature_0"] + X["feature_1"] > 0).astype(int))
    return X, y


@pytest.fixture
def regression_data():
    """Generate synthetic regression data."""
    np.random.seed(42)
    n = 500
    X = pd.DataFrame({f"feature_{i}": np.random.randn(n) for i in range(20)})
    y = pd.Series(3 * X["feature_0"] + 2 * X["feature_1"] + np.random.randn(n))
    return X, y


class TestModelTypeCategory:
    """Test model type classification."""

    def test_classifier_types(self):
        assert get_model_type_category("xgboost_classifier") == "classifier"
        assert get_model_type_category("lightgbm_classifier") == "classifier"

    def test_regressor_types(self):
        assert get_model_type_category("xgboost_regressor") == "regressor"
        assert get_model_type_category("lightgbm_regressor") == "regressor"

    def test_clusterer_types(self):
        assert get_model_type_category("sklearn_kmeans") == "clusterer"

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown model type"):
            get_model_type_category("unknown_model")


class TestModelTraining:
    """Test model training functionality."""

    def test_xgboost_classifier_trains(self, classification_data):
        """XGBoost classifier should train and produce valid predictions."""
        import xgboost as xgb

        X, y = classification_data
        model = xgb.XGBClassifier(
            max_depth=3,
            n_estimators=50,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
        )
        model.fit(X, y)

        preds = model.predict(X)
        probs = model.predict_proba(X)

        assert len(preds) == len(X)
        assert set(preds).issubset({0, 1})
        assert probs.shape == (len(X), 2)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_xgboost_regressor_trains(self, regression_data):
        """XGBoost regressor should train and produce continuous predictions."""
        import xgboost as xgb

        X, y = regression_data
        model = xgb.XGBRegressor(
            max_depth=3,
            n_estimators=50,
            verbosity=0,
        )
        model.fit(X, y)

        preds = model.predict(X)
        assert len(preds) == len(X)
        assert not np.isnan(preds).any()

    def test_lightgbm_classifier_trains(self, classification_data):
        """LightGBM classifier should train successfully."""
        import lightgbm as lgb

        X, y = classification_data
        model = lgb.LGBMClassifier(
            max_depth=3,
            n_estimators=50,
            verbose=-1,
        )
        model.fit(X, y)

        preds = model.predict(X)
        assert len(preds) == len(X)

    def test_kmeans_clusters(self, classification_data):
        """KMeans should produce cluster assignments."""
        from sklearn.cluster import KMeans

        X, _ = classification_data
        model = KMeans(n_clusters=3, random_state=42, n_init=10)
        model.fit(X)

        labels = model.labels_
        assert len(labels) == len(X)
        assert len(set(labels)) == 3

    def test_mlflow_logging(self, classification_data, tmp_path):
        """MLflow should log parameters and metrics."""
        import xgboost as xgb

        X, y = classification_data
        # MLflow v3.14+ requires a database backend; file-store is deprecated
        db_path = tmp_path / "mlflow.db"
        tracking_uri = f"sqlite:///{db_path}"
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("test-experiment")

        with mlflow.start_run() as run:
            model = xgb.XGBClassifier(
                max_depth=3,
                n_estimators=50,
                use_label_encoder=False,
                eval_metric="logloss",
                verbosity=0,
            )
            model.fit(X, y)

            mlflow.log_param("max_depth", 3)
            mlflow.log_metric("accuracy", 0.95)
            mlflow.xgboost.log_model(model, artifact_path="model")

        # Verify run was logged
        client = mlflow.tracking.MlflowClient(tracking_uri)
        run_data = client.get_run(run.info.run_id)

        assert run_data.data.params["max_depth"] == "3"
        assert float(run_data.data.metrics["accuracy"]) == 0.95

    def test_cross_validation_scores(self, classification_data):
        """Cross-validation should produce reasonable AUC scores."""
        import xgboost as xgb
        from sklearn.model_selection import cross_val_score

        X, y = classification_data
        model = xgb.XGBClassifier(
            max_depth=3,
            n_estimators=50,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
        )

        scores = cross_val_score(model, X, y, cv=3, scoring="roc_auc")
        assert len(scores) == 3
        assert all(0 <= s <= 1 for s in scores)
        assert np.mean(scores) > 0.5  # Better than random


class TestChampionChallenger:
    """Test champion/challenger comparison logic."""

    def test_model_meets_threshold(self):
        """Model meeting minimum threshold should be promoted."""
        metrics = {"roc_auc": 0.85, "f1": 0.80}
        threshold = {"roc_auc_min": 0.65}

        meets = all(metrics.get(k.replace("_min", ""), 0) >= v for k, v in threshold.items())
        assert meets is True

    def test_model_below_threshold(self):
        """Model below minimum threshold should be rejected."""
        metrics = {"roc_auc": 0.55, "f1": 0.50}
        threshold = {"roc_auc_min": 0.65}

        meets = all(metrics.get(k.replace("_min", ""), 0) >= v for k, v in threshold.items())
        assert meets is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
