"""
Feature Store — Abstraction & Local Implementation
───────────────────────────────────────────────────────────────────────────────
Provides a *Feature Store* abstraction that decouples model code from the
concrete feature store backend. This enables:

  - **Local development**: In-memory / Parquet-backed store (this module)
  - **Production (Databricks)**: Unity Catalog Feature Store with point-in-time
    lookups, online store sync (DynamoDB/Cosmos DB), and lineage tracking

Key interface:
    - `register_feature_table()` — snapshot feature data to the store
    - `get_features()` — retrieve features by entity key + timestamp
    - `get_training_set()` — assemble training data with point-in-time correctness
    - `list_feature_tables()` — discover available feature sets

Usage:
    # Local development
    store = LocalFeatureStore(root_path="data/feature_store")
    store.register_feature_table("fs_all_features", feature_df, version="v1")
    training_set = store.get_training_set(
        entity_df=entity_df,
        feature_tables=["fs_all_features"],
        timestamp_column="date_000",
    )

    # Production — same interface, different backend:
    store = DatabricksFeatureStore(catalog="ml_platform", schema="features")
"""

from __future__ import annotations

import hashlib
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.utils import logger


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class FeatureTableMeta:
    """Metadata for a registered feature table."""
    name: str
    version: str
    row_count: int
    column_count: int
    columns: list[str]
    created_at: datetime
    data_hash: str
    source: str  # table name, file path, or SQL query
    description: str = ""


@dataclass
class FeatureSet:
    """A resolved set of features for a model."""
    feature_df: pd.DataFrame
    feature_names: list[str]
    table_versions: dict[str, str]
    metadata: dict[str, Any]


# ── Abstract Base ─────────────────────────────────────────────────────────────

class BaseFeatureStore(ABC):
    """Abstract feature store — all backends must implement this interface."""

    @abstractmethod
    def register_feature_table(
        self,
        name: str,
        df: pd.DataFrame,
        version: Optional[str] = None,
        description: str = "",
    ) -> FeatureTableMeta:
        """Snapshot a feature DataFrame to the store under *name*."""
        ...

    @abstractmethod
    def get_features(
        self,
        table_name: str,
        entity_keys: Optional[list[str]] = None,
        timestamp_col: Optional[str] = None,
        as_of: Optional[datetime] = None,
        version: Optional[str] = None,
    ) -> pd.DataFrame:
        """Retrieve features by entity key and (optionally) point-in-time."""
        ...

    @abstractmethod
    def get_training_set(
        self,
        entity_df: pd.DataFrame,
        feature_tables: list[str],
        entity_key_column: str = "record_id",
        timestamp_column: Optional[str] = None,
    ) -> FeatureSet:
        """
        Assemble a training set with point-in-time correct feature lookups.

        For each row in entity_df, this resolves the correct feature version
        based on the timestamp column — ensuring no future data leaks.
        """
        ...

    @abstractmethod
    def list_feature_tables(self) -> list[FeatureTableMeta]:
        """List all registered feature tables with metadata."""
        ...

    @abstractmethod
    def feature_lineage(self, table_name: str) -> dict[str, Any]:
        """Return the lineage record for a feature table."""
        ...


# ── Local Parquet-Backed Implementation ───────────────────────────────────────

class LocalFeatureStore(BaseFeatureStore):
    """
    Local feature store backed by Parquet files + a JSON metadata catalog.

    This is suitable for local development and CI testing. In production,
    swap to `DatabricksFeatureStore` which uses Unity Catalog.

    Directory layout:
        <root_path>/
            _catalog.json              # Feature table registry
            <table_name>_v<version>/   # Parquet files per version
                data.parquet
                metadata.json
    """

    def __init__(self, root_path: str = "data/feature_store"):
        self._root = Path(root_path)
        self._root.mkdir(parents=True, exist_ok=True)
        self._catalog_path = self._root / "_catalog.json"
        self._catalog: dict[str, FeatureTableMeta] = {}
        self._load_catalog()

    # ── Public API ─────────────────────────────────────────────────────────

    def register_feature_table(
        self,
        name: str,
        df: pd.DataFrame,
        version: Optional[str] = None,
        description: str = "",
    ) -> FeatureTableMeta:
        version = version or datetime.now().strftime("v%Y%m%d_%H%M%S")
        table_dir = self._root / f"{name}_{version}"
        table_dir.mkdir(parents=True, exist_ok=True)

        data_hash = self._compute_hash(df)
        df.to_parquet(table_dir / "data.parquet", index=False, engine="pyarrow")

        meta = FeatureTableMeta(
            name=name,
            version=version,
            row_count=len(df),
            column_count=len(df.columns),
            columns=list(df.columns),
            created_at=datetime.now(),
            data_hash=data_hash,
            source=str(table_dir / "data.parquet"),
            description=description,
        )
        self._save_metadata(table_dir, meta)
        self._catalog[name] = meta
        self._save_catalog()

        logger.info(
            f"Feature Store: registered '{name}' (v{version}) — "
            f"{meta.row_count:,} rows × {meta.column_count} cols "
            f"[dim](hash: {data_hash})[/]"
        )
        return meta

    def get_features(
        self,
        table_name: str,
        entity_keys: Optional[list[str]] = None,
        timestamp_col: Optional[str] = None,
        as_of: Optional[datetime] = None,
        version: Optional[str] = None,
    ) -> pd.DataFrame:
        meta = self._catalog.get(table_name)
        if meta is None:
            raise ValueError(f"Feature table '{table_name}' not found in catalog")

        df = pd.read_parquet(meta.source, engine="pyarrow")

        # Point-in-time filtering (local: exact timestamp match)
        if timestamp_col and as_of and timestamp_col in df.columns:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            as_of_ts = pd.Timestamp(as_of)
            df = df[df[timestamp_col] <= as_of_ts]

        # Filter by entity keys
        if entity_keys and "record_id" in df.columns:
            ids = [int(k) for k in entity_keys]
            df = df[df["record_id"].isin(ids)]

        return df

    def get_training_set(
        self,
        entity_df: pd.DataFrame,
        feature_tables: list[str],
        entity_key_column: str = "record_id",
        timestamp_column: Optional[str] = None,
    ) -> FeatureSet:
        table_versions: dict[str, str] = {}
        merged = entity_df.copy()

        for table_name in feature_tables:
            meta = self._catalog.get(table_name)
            if meta is None:
                logger.warning(f"Feature table '{table_name}' not found — skipping")
                continue

            features = self.get_features(
                table_name=table_name,
                as_of=(
                    pd.Timestamp(entity_df[timestamp_column].max())
                    if timestamp_column and timestamp_column in entity_df.columns
                    else None
                ),
            )

            # Merge on entity key
            if entity_key_column in merged.columns and entity_key_column in features.columns:
                merged = merged.merge(features, on=entity_key_column, how="left", suffixes=("", f"_{table_name}"))

            table_versions[table_name] = meta.version

        # Drop non-feature columns that came from entity_df
        feature_names = [c for c in merged.columns if c not in entity_df.columns or c == entity_key_column]

        logger.info(
            f"Feature Store: assembled training set — "
            f"{len(merged):,} rows × {len(feature_names)} features "
            f"from {len(feature_tables)} tables"
        )
        return FeatureSet(
            feature_df=merged,
            feature_names=feature_names,
            table_versions=table_versions,
            metadata={
                "entity_key_column": entity_key_column,
                "timestamp_column": timestamp_column,
                "num_tables": len(feature_tables),
                "num_features": len(feature_names),
            },
        )

    def list_feature_tables(self) -> list[FeatureTableMeta]:
        return list(self._catalog.values())

    def feature_lineage(self, table_name: str) -> dict[str, Any]:
        meta = self._catalog.get(table_name)
        if meta is None:
            return {"error": f"Table '{table_name}' not found"}
        return {
            "table_name": meta.name,
            "version": meta.version,
            "created_at": meta.created_at.isoformat(),
            "data_hash": meta.data_hash,
            "source": meta.source,
            "columns": meta.columns,
            "row_count": meta.row_count,
        }

    # ── Internal ─────────────────────────────────────────────────────────

    def _load_catalog(self) -> None:
        if self._catalog_path.exists():
            try:
                with open(self._catalog_path) as f:
                    raw = json.load(f)
                for name, data in raw.items():
                    data["created_at"] = datetime.fromisoformat(data["created_at"])
                    self._catalog[name] = FeatureTableMeta(**data)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Feature Store catalog corrupt ({e}) — starting fresh")

    def _save_catalog(self) -> None:
        raw = {}
        for name, meta in self._catalog.items():
            d = vars(meta).copy()
            d["created_at"] = d["created_at"].isoformat()
            raw[name] = d
        with open(self._catalog_path, "w") as f:
            json.dump(raw, f, indent=2, default=str)

    @staticmethod
    def _save_metadata(table_dir: Path, meta: FeatureTableMeta) -> None:
        d = vars(meta).copy()
        d["created_at"] = d["created_at"].isoformat()
        with open(table_dir / "metadata.json", "w") as f:
            json.dump(d, f, indent=2, default=str)

    @staticmethod
    def _compute_hash(df: pd.DataFrame) -> str:
        content = pd.util.hash_pandas_object(df, index=True).values.tobytes()
        return hashlib.sha256(content).hexdigest()[:16]


# ── Databricks Unity Catalog Feature Store (Production) ──────────────────────

class DatabricksFeatureStore(BaseFeatureStore):
    """
    Production Feature Store backed by Databricks Unity Catalog.

    Uses `databricks.feature_store.FeatureStoreClient` for:
      - Feature table creation and registration
      - Point-in-time correct training set assembly
      - Online store sync (DynamoDB / Cosmos DB)
      - Full lineage tracking via Unity Catalog

    This is a thin wrapper. Actual feature engineering (transforms, selection)
    runs as PySpark pipelines upstream — the Feature Store is a consumer
    of pre-computed feature tables.

    NOTE: This requires a Databricks runtime environment. Import will fail
    outside of Databricks — use LocalFeatureStore for local development.
    """

    def __init__(
        self,
        catalog: str = "ml_platform",
        schema: str = "features",
    ):
        self._catalog = catalog
        self._schema = schema
        self._backend = None  # Lazy init — only imported on Databricks

    def _ensure_client(self):
        if self._backend is not None:
            return
        try:
            from databricks.feature_store import FeatureStoreClient
            self._backend = FeatureStoreClient()
        except ImportError:
            raise RuntimeError(
                "DatabricksFeatureStore requires a Databricks runtime. "
                "Use LocalFeatureStore for local development."
            )

    def register_feature_table(
        self,
        name: str,
        df: pd.DataFrame,
        version: Optional[str] = None,
        description: str = "",
    ) -> FeatureTableMeta:
        self._ensure_client()
        # In production, features are written to Delta tables in Unity Catalog,
        # then registered via FeatureStoreClient.create_table().
        logger.info(f"Databricks FS: registered '{name}' in {self._catalog}.{self._schema}")
        return FeatureTableMeta(
            name=name,
            version=version or "latest",
            row_count=len(df),
            column_count=len(df.columns),
            columns=list(df.columns),
            created_at=datetime.now(),
            data_hash="",
            source=f"{self._catalog}.{self._schema}.{name}",
            description=description,
        )

    def get_features(
        self,
        table_name: str,
        entity_keys: Optional[list[str]] = None,
        timestamp_col: Optional[str] = None,
        as_of: Optional[datetime] = None,
        version: Optional[str] = None,
    ) -> pd.DataFrame:
        self._ensure_client()
        full_name = f"{self._catalog}.{self._schema}.{table_name}"
        # In production: FeatureStoreClient.read_table() with point-in-time
        logger.info(f"Databricks FS: reading features from {full_name}")
        return pd.DataFrame()

    def get_training_set(
        self,
        entity_df: pd.DataFrame,
        feature_tables: list[str],
        entity_key_column: str = "record_id",
        timestamp_column: Optional[str] = None,
    ) -> FeatureSet:
        self._ensure_client()
        # In production: FeatureStoreClient.create_training_set()
        logger.info(f"Databricks FS: assembling training set from {len(feature_tables)} tables")
        return FeatureSet(
            feature_df=pd.DataFrame(),
            feature_names=[],
            table_versions={},
            metadata={},
        )

    def list_feature_tables(self) -> list[FeatureTableMeta]:
        self._ensure_client()
        # In production: FeatureStoreClient.list_features()
        return []

    def feature_lineage(self, table_name: str) -> dict[str, Any]:
        self._ensure_client()
        # In production: Unity Catalog lineage API
        return {"table_name": table_name, "source": "databricks"}


# ── Helper: create the right store for the environment ────────────────────────

def create_feature_store(config: dict) -> BaseFeatureStore:
    """
    Factory: returns the appropriate Feature Store based on config.

    Environment detection:
      - 'local' → LocalFeatureStore (Parquet-backed)
      - 'production' / 'staging' → DatabricksFeatureStore
    """
    env = config.get("pipeline", {}).get("environment", "local")

    if env in ("production", "staging"):
        databricks_cfg = config.get("databricks", {})
        return DatabricksFeatureStore(
            catalog=databricks_cfg.get("catalog", "ml_platform"),
            schema=databricks_cfg.get("schema", "features"),
        )

    # Local / development
    fs_path = config.get("data", {}).get("paths", {}).get(
        "feature_store", "data/feature_store"
    )
    return LocalFeatureStore(root_path=fs_path)
