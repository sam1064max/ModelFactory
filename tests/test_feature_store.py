"""
Unit Tests — Feature Store
───────────────────────────────────────────────────────────────────────────────
Tests the LocalFeatureStore implementation for:
  - Table registration and catalog persistence
  - Feature retrieval by entity keys
  - Training set assembly with point-in-time correctness
  - Lineage tracking
  - Cross-session catalog persistence (JSON)
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.feature_store import (
    LocalFeatureStore,
    DatabricksFeatureStore,
    create_feature_store,
)


@pytest.fixture
def store():
    """Create a LocalFeatureStore in a temp directory."""
    with tempfile.TemporaryDirectory() as tmp:
        yield LocalFeatureStore(root_path=str(Path(tmp) / "feature_store"))


@pytest.fixture
def sample_features():
    """Generate a sample feature DataFrame."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame(
        {
            "record_id": range(n),
            "feature_alpha": np.random.randn(n),
            "feature_beta": np.random.uniform(0, 100, n),
            "feature_gamma": np.random.choice(["A", "B", "C"], n),
            "date_ref": pd.date_range("2024-01-01", periods=n, freq="D"),
        }
    )


class TestLocalFeatureStore:
    """Test suite for LocalFeatureStore."""

    def test_register_and_retrieve(self, store, sample_features):
        """Registered features should be retrievable."""
        meta = store.register_feature_table(
            "fs_test",
            sample_features,
            version="v1",
            description="Test feature set",
        )
        assert meta.name == "fs_test"
        assert meta.version == "v1"
        assert meta.row_count == len(sample_features)
        assert meta.column_count == len(sample_features.columns)

        retrieved = store.get_features("fs_test")
        assert len(retrieved) == len(sample_features)

    def test_register_generates_version(self, store, sample_features):
        """Omitting version should auto-generate one."""
        meta = store.register_feature_table("fs_auto", sample_features)
        assert meta.version.startswith("v")

    def test_get_features_filters_by_entity_keys(self, store, sample_features):
        """Entity key filtering should return only matching records."""
        store.register_feature_table("fs_keys", sample_features, version="v1")

        result = store.get_features("fs_keys", entity_keys=["0", "1", "2"])
        assert len(result) == 3

    def test_get_features_missing_table(self, store):
        """Requesting a missing table should raise ValueError."""
        with pytest.raises(ValueError, match="not found"):
            store.get_features("nonexistent")

    def test_get_training_set_assembles_features(self, store, sample_features):
        """Training set should merge entity keys with feature tables."""
        store.register_feature_table("fs_main", sample_features, version="v1")

        entity_df = pd.DataFrame({"record_id": [0, 1, 2, 3, 4]})
        training_set = store.get_training_set(
            entity_df=entity_df,
            feature_tables=["fs_main"],
        )

        assert len(training_set.feature_df) == 5
        assert "feature_alpha" in training_set.feature_names

    def test_list_feature_tables(self, store, sample_features):
        """list_feature_tables should return all registered tables."""
        store.register_feature_table("fs_a", sample_features, version="v1")
        store.register_feature_table("fs_b", sample_features, version="v1")

        tables = store.list_feature_tables()
        assert len(tables) == 2
        names = [t.name for t in tables]
        assert "fs_a" in names
        assert "fs_b" in names

    def test_feature_lineage(self, store, sample_features):
        """Lineage should return metadata about the feature table."""
        store.register_feature_table("fs_lineage", sample_features, version="v2")

        lineage = store.feature_lineage("fs_lineage")
        assert lineage["table_name"] == "fs_lineage"
        assert lineage["version"] == "v2"
        assert "data_hash" in lineage
        assert "source" in lineage

    def test_catalog_persists_across_sessions(self, sample_features):
        """The JSON catalog should persist when creating a new store instance."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "feature_store"

            # First session: register a table
            store1 = LocalFeatureStore(root_path=str(root))
            store1.register_feature_table("fs_persist", sample_features, version="v1")

            # Second session: new instance, same directory
            store2 = LocalFeatureStore(root_path=str(root))
            tables = store2.list_feature_tables()
            assert len(tables) == 1
            assert tables[0].name == "fs_persist"

    def test_training_set_tracks_table_versions(self, store, sample_features):
        """Training set should record which table versions were used."""
        store.register_feature_table("fs_v1", sample_features, version="v1")

        entity_df = pd.DataFrame({"record_id": [0, 1]})
        training_set = store.get_training_set(
            entity_df=entity_df,
            feature_tables=["fs_v1"],
        )

        assert training_set.table_versions["fs_v1"] == "v1"

    def test_create_feature_store_factory_local(self):
        """create_feature_store should return LocalFeatureStore for 'local' env."""
        config = {"pipeline": {"environment": "local"}}
        fs = create_feature_store(config)
        assert isinstance(fs, LocalFeatureStore)

    def test_create_feature_store_factory_production(self):
        """create_feature_store should return DatabricksFeatureStore for prod."""
        config = {
            "pipeline": {"environment": "production"},
            "databricks": {"catalog": "ml_platform", "schema": "production"},
        }
        fs = create_feature_store(config)
        assert isinstance(fs, DatabricksFeatureStore)
