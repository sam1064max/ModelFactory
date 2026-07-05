"""
Data Ingestion Module
──────────────────────────────────────────────────────────────────────────────
Generates synthetic training and inference data that simulates real-world
enterprise datasets with numeric, categorical, temporal, and text features.

In production (Databricks), this would be replaced by:
  - Auto Loader for cloud storage ingestion
  - Delta Live Tables for quality-validated ETL
  - JDBC connectors for relational sources
  - Kafka/Kinesis for streaming sources
"""

import string
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from src.utils import (
    ensure_directories,
    logger,
    save_parquet,
    timer,
)


def generate_synthetic_data(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic training and inference datasets.

    Returns:
        tuple: (training_df, inference_df)
    """
    with timer("Data Ingestion — Synthetic Data Generation"):
        ensure_directories(config)
        data_config = config["data"]["synthetic"]

        # Generate training data
        train_df = _generate_dataset(
            n_rows=data_config["training_rows"],
            n_numeric=data_config["num_numeric_features"],
            n_categorical=data_config["num_categorical_features"],
            n_temporal=data_config["num_temporal_features"],
            n_text=data_config["num_text_features"],
            seed=data_config["random_seed"],
            include_targets=True,
            dataset_name="training",
        )

        # Generate inference data (no targets)
        inference_df = _generate_dataset(
            n_rows=data_config["inference_rows"],
            n_numeric=data_config["num_numeric_features"],
            n_categorical=data_config["num_categorical_features"],
            n_temporal=data_config["num_temporal_features"],
            n_text=data_config["num_text_features"],
            seed=data_config["random_seed"] + 1,
            include_targets=False,
            dataset_name="inference",
        )

        # Save as Parquet (Bronze layer)
        paths = config["data"]["paths"]
        save_parquet(train_df, f"{paths['raw_data']}training_data.parquet", "Training data")
        save_parquet(
            inference_df,
            f"{paths['inference_input']}inference_universe.parquet",
            "Inference universe",
        )

        # Validate data quality
        _validate_data_quality(train_df, "training")
        _validate_data_quality(inference_df, "inference")

        return train_df, inference_df


def _generate_dataset(
    n_rows: int,
    n_numeric: int,
    n_categorical: int,
    n_temporal: int,
    n_text: int,
    seed: int,
    include_targets: bool,
    dataset_name: str,
) -> pd.DataFrame:
    """Generate a synthetic dataset with multiple feature types."""
    rng = np.random.RandomState(seed)
    data: dict[str, Any] = {}

    # Record ID
    data["record_id"] = np.arange(n_rows)

    # ── Numeric Features ─────────────────────────────────────────────────
    for i in range(n_numeric):
        distribution = rng.choice(["normal", "lognormal", "uniform", "exponential"])
        if distribution == "normal":
            data[f"num_{i:03d}"] = rng.normal(
                loc=rng.uniform(-10, 10), scale=rng.uniform(0.5, 5), size=n_rows
            )
        elif distribution == "lognormal":
            data[f"num_{i:03d}"] = rng.lognormal(
                mean=rng.uniform(0, 2), sigma=rng.uniform(0.3, 1.5), size=n_rows
            )
        elif distribution == "uniform":
            low = rng.uniform(-100, 0)
            data[f"num_{i:03d}"] = rng.uniform(low, low + rng.uniform(10, 200), size=n_rows)
        else:
            data[f"num_{i:03d}"] = rng.exponential(scale=rng.uniform(1, 10), size=n_rows)

    # Inject some nulls (realistic data quality)
    null_features = rng.choice(
        [f"num_{i:03d}" for i in range(n_numeric)],
        size=max(1, n_numeric // 5),
        replace=False,
    )
    for feat in null_features:
        null_mask = rng.random(n_rows) < rng.uniform(0.01, 0.05)
        data[feat] = np.where(null_mask, np.nan, data[feat])

    # ── Categorical Features ─────────────────────────────────────────────
    categories_pool = {
        "geography": ["US", "EU", "APAC", "LATAM", "MEA"],
        "segment": ["retail", "wholesale", "enterprise", "smb"],
        "channel": ["online", "store", "phone", "partner", "direct"],
        "product_line": ["A", "B", "C", "D", "E", "F"],
        "risk_tier": ["low", "medium", "high", "very_high"],
        "customer_type": ["new", "returning", "loyal", "dormant"],
    }

    for i in range(n_categorical):
        if i < len(categories_pool):
            name, categories = list(categories_pool.items())[i]
            data[f"cat_{name}"] = rng.choice(categories, size=n_rows)
        else:
            n_cats = rng.randint(3, 20)
            cats = [
                "".join(rng.choice(list(string.ascii_uppercase), size=3)) for _ in range(n_cats)
            ]
            data[f"cat_{i:03d}"] = rng.choice(cats, size=n_rows)

    # ── Temporal Features ────────────────────────────────────────────────
    base_date = datetime(2024, 1, 1)
    for i in range(n_temporal):
        days_offset = rng.randint(0, 730, size=n_rows)
        dates = [base_date + timedelta(days=int(d)) for d in days_offset]
        data[f"date_{i:03d}"] = dates

    # ── Text Features (simplified as length/hash proxies) ────────────────
    for i in range(n_text):
        lengths = rng.randint(5, 200, size=n_rows)
        data[f"text_len_{i:03d}"] = lengths
        data[f"text_words_{i:03d}"] = (lengths / rng.uniform(4, 6, size=n_rows)).astype(int)

    # ── Target Variables ─────────────────────────────────────────────────
    if include_targets:
        # Binary target (classification) — correlated with some features
        logit = (
            0.3
            * (data["num_000"] - np.nanmean(data["num_000"]))
            / (np.nanstd(data["num_000"]) + 1e-8)
            + 0.2
            * (data["num_001"] - np.nanmean(data["num_001"]))
            / (np.nanstd(data["num_001"]) + 1e-8)
            + rng.normal(0, 1, size=n_rows)
        )
        prob = 1 / (1 + np.exp(-logit))
        data["target_binary"] = (rng.random(n_rows) < prob).astype(int)

        # Continuous target (regression) — correlated with features
        data["target_continuous"] = (
            5.0 * data["num_000"]
            + 3.0 * np.nan_to_num(data["num_002"])
            - 2.0 * np.nan_to_num(data["num_003"])
            + rng.normal(0, 10, size=n_rows)
        )

    df = pd.DataFrame(data)
    logger.info(
        f"Generated {dataset_name} dataset: "
        f"{df.shape[0]:,} rows × {df.shape[1]} cols "
        f"[dim](targets: {include_targets})[/]"
    )
    return df


def _validate_data_quality(df: pd.DataFrame, dataset_name: str) -> None:
    """Run basic data quality checks, simulating Delta Live Tables expectations."""
    checks_passed = 0
    checks_failed = 0

    # Check 1: No duplicate record IDs
    if df["record_id"].nunique() == len(df):
        checks_passed += 1
    else:
        checks_failed += 1
        logger.warning(f"[{dataset_name}] Duplicate record_ids detected!")

    # Check 2: Null rates within acceptable range
    null_rates = df.isnull().mean()
    high_null_cols = null_rates[null_rates > 0.1].index.tolist()
    if not high_null_cols:
        checks_passed += 1
    else:
        checks_failed += 1
        logger.warning(f"[{dataset_name}] High null rate columns: {high_null_cols}")

    # Check 3: No constant columns
    constant_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns if df[c].nunique() <= 1
    ]
    if not constant_cols:
        checks_passed += 1
    else:
        checks_failed += 1
        logger.warning(f"[{dataset_name}] Constant columns: {constant_cols}")

    # Check 4: Row count within expected range
    if len(df) > 0:
        checks_passed += 1
    else:
        checks_failed += 1
        logger.error(f"[{dataset_name}] Empty DataFrame!")

    status = "✅" if checks_failed == 0 else "⚠️"
    logger.info(
        f"{status} Data quality [{dataset_name}]: {checks_passed} passed, {checks_failed} failed"
    )
