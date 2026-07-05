"""
Shared Utilities & Logging
──────────────────────────────────────────────────────────────────────────────
Provides configuration loading, structured logging, timing utilities, and
data hashing for reproducibility tracking throughout the pipeline.
"""

import hashlib
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from rich.console import Console
from rich.logging import RichHandler

# ── Rich Console ─────────────────────────────────────────────────────────────
console = Console()


# ── Logging Setup ────────────────────────────────────────────────────────────
def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Create a structured logger with Rich formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = RichHandler(
            console=console,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper()))
    return logger


logger = setup_logger("mlops_pipeline")


# ── Configuration Loading ────────────────────────────────────────────────────
def load_config(config_path: str = "config/pipeline_config.yaml") -> dict[str, Any]:
    """Load pipeline configuration from YAML file."""
    path = _resolve_path(config_path)
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config from [bold cyan]{path}[/]")
    return config  # type: ignore[no-any-return]


def load_model_registry(
    registry_path: str = "config/model_registry.yaml",
) -> dict[str, Any]:
    """Load model registry configuration from YAML file."""
    path = _resolve_path(registry_path)
    with open(path, "r") as f:
        registry = yaml.safe_load(f)
    num_models = len(registry.get("models", []))
    logger.info(f"Loaded model registry from [bold cyan]{path}[/] ({num_models} models)")
    return registry  # type: ignore[no-any-return]


def _resolve_path(relative_path: str) -> Path:
    """Resolve a path relative to the mlops_pipeline directory."""
    # Walk up from this file to find the mlops_pipeline root
    base = Path(__file__).resolve().parent.parent
    candidate = base / relative_path
    if candidate.exists():
        return candidate
    # Fallback: try current working directory
    return Path(relative_path)


# ── Directory Management ─────────────────────────────────────────────────────
def ensure_directories(config: dict) -> None:
    """Create all data directories specified in the config."""
    paths = config.get("data", {}).get("paths", {})
    base = Path(__file__).resolve().parent.parent
    for key, rel_path in paths.items():
        full_path = base / rel_path
        full_path.mkdir(parents=True, exist_ok=True)
    logger.info("All data directories ensured")


# ── Timing Utilities ─────────────────────────────────────────────────────────
@contextmanager
def timer(task_name: str):
    """Context manager for timing pipeline stages."""
    start = time.time()
    logger.info(f"[bold yellow]▶ Starting:[/] {task_name}")
    try:
        yield
    finally:
        elapsed = time.time() - start
        if elapsed < 60:
            time_str = f"{elapsed:.1f}s"
        elif elapsed < 3600:
            time_str = f"{elapsed / 60:.1f}m"
        else:
            time_str = f"{elapsed / 3600:.1f}h"
        logger.info(f"[bold green]✓ Completed:[/] {task_name} in {time_str}")


# ── Data Hashing ─────────────────────────────────────────────────────────────
def compute_data_hash(df: pd.DataFrame) -> str:
    """Compute a deterministic hash of a DataFrame for versioning."""
    content = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.sha256(content).hexdigest()[:16]


# ── Parquet I/O ──────────────────────────────────────────────────────────────
def save_parquet(df: pd.DataFrame, path: str, description: str = "") -> str:
    """Save DataFrame as Parquet with logging."""
    full_path = _ensure_output_path(path)
    df.to_parquet(full_path, index=False, engine="pyarrow")
    size_mb = os.path.getsize(full_path) / (1024 * 1024)
    data_hash = compute_data_hash(df)
    logger.info(
        f"Saved {description or 'data'}: {df.shape[0]:,} rows × {df.shape[1]} cols "
        f"({size_mb:.1f} MB) → [cyan]{full_path}[/] [dim](hash: {data_hash})[/]"
    )
    return data_hash


def load_parquet(path: str, description: str = "") -> pd.DataFrame:
    """Load DataFrame from Parquet with logging."""
    full_path = _resolve_data_path(path)
    df = pd.read_parquet(full_path, engine="pyarrow")
    logger.info(
        f"Loaded {description or 'data'}: {df.shape[0]:,} rows × {df.shape[1]} cols "
        f"from [cyan]{full_path}[/]"
    )
    return df


def _ensure_output_path(relative_path: str) -> Path:
    """Resolve and ensure the output path exists."""
    base = Path(__file__).resolve().parent.parent
    full_path = base / relative_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    return full_path


def _resolve_data_path(relative_path: str) -> Path:
    """Resolve a data path relative to the project root."""
    base = Path(__file__).resolve().parent.parent
    return base / relative_path


# ── Model Config Helpers ─────────────────────────────────────────────────────
def get_model_type_category(model_type: str) -> str:
    """Classify model type into category: classifier, regressor, or clusterer."""
    if "classifier" in model_type or "clf" in model_type:
        return "classifier"
    elif "regressor" in model_type or "reg" in model_type:
        return "regressor"
    elif "kmeans" in model_type or "cluster" in model_type or "clust" in model_type:
        return "clusterer"
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def flatten_hyperparameter_space(space: dict[str, list]) -> list[dict[str, Any]]:
    """
    Generate a flat list of hyperparameter combinations from a space definition.
    For HPO, we sample from this space rather than grid-searching.
    """
    import itertools

    keys = list(space.keys())
    values = list(space.values())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return combinations
