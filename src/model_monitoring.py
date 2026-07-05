"""
Model Monitoring Module
──────────────────────────────────────────────────────────────────────────────
Implements drift detection and data quality monitoring:
  - Population Stability Index (PSI) for feature drift
  - Kolmogorov-Smirnov test for distribution shift
  - Prediction distribution analysis
  - Data quality metrics

In production (Databricks), this would use:
  - Databricks Lakehouse Monitoring for automatic drift detection
  - Evidently AI for custom statistical tests and reporting
  - SQL-based drift queries over Delta Lake tables
  - Alerting via PagerDuty/Slack webhooks
"""

import json
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats

from src.utils import logger, save_parquet, timer

# Try importing evidently (optional dependency for rich reports)
try:
    from evidently import ColumnMapping
    from evidently.metric_preset import (
        DataDriftPreset,
        DataQualityPreset,
        TargetDriftPreset,
    )
    from evidently.report import Report

    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    logger.warning(
        "Evidently not installed — using built-in drift detection. "
        "Install with: pip install evidently"
    )


def run_monitoring(
    train_features: pd.DataFrame,
    inference_features: pd.DataFrame,
    predictions: dict[str, pd.DataFrame],
    config: dict,
) -> dict:
    """
    Run comprehensive monitoring pipeline.

    Args:
        train_features: Training feature distributions (reference)
        inference_features: Inference feature distributions (current)
        predictions: Model predictions from batch inference
        config: Pipeline configuration

    Returns:
        dict: Monitoring results and alerts
    """
    monitoring_config = config["monitoring"]
    output_path = config["data"]["paths"]["monitoring_reports"]

    with timer("Monitoring & Observability"):
        results: dict[str, Any] = {
            "feature_drift": {},
            "prediction_drift": {},
            "data_quality": {},
            "alerts": [],
        }

        # ── 1. Feature Drift Detection ───────────────────────────────────
        logger.info("[bold cyan]1. Feature Drift Detection[/]")
        feature_drift = _compute_feature_drift(
            reference=train_features,
            current=inference_features,
            psi_threshold=monitoring_config["drift_detection"]["psi_threshold"],
            ks_threshold=monitoring_config["drift_detection"]["ks_threshold"],
        )
        results["feature_drift"] = feature_drift

        # Count drifted features
        drifted = sum(1 for v in feature_drift.values() if v.get("is_drifted", False))
        total = len(feature_drift)
        drift_pct = (drifted / total * 100) if total > 0 else 0

        logger.info(f"  Feature drift: {drifted}/{total} features drifted ({drift_pct:.1f}%)")

        if drift_pct > 50:
            results["alerts"].append(
                {
                    "severity": "P1",
                    "type": "feature_drift",
                    "message": f"{drift_pct:.1f}% of features show significant drift",
                }
            )
        elif drift_pct > 20:
            results["alerts"].append(
                {
                    "severity": "P2",
                    "type": "feature_drift",
                    "message": f"{drift_pct:.1f}% of features show drift",
                }
            )

        # ── 2. Prediction Distribution Analysis ─────────────────────────
        logger.info("[bold cyan]2. Prediction Distribution Analysis[/]")
        for model_id, preds_df in predictions.items():
            pred_stats = _analyze_prediction_distribution(preds_df, model_id)
            results["prediction_drift"][model_id] = pred_stats

        # ── 3. Data Quality Metrics ──────────────────────────────────────
        logger.info("[bold cyan]3. Data Quality Metrics[/]")
        quality = _compute_data_quality(
            inference_features,
            null_threshold=monitoring_config["data_quality"]["null_threshold"],
        )
        results["data_quality"] = quality

        quality_issues = sum(1 for v in quality.values() if v.get("has_issue", False))
        if quality_issues > 0:
            results["alerts"].append(
                {
                    "severity": "P3",
                    "type": "data_quality",
                    "message": f"{quality_issues} features have data quality issues",
                }
            )

        # ── 4. Evidently Report (if available) ───────────────────────────
        if EVIDENTLY_AVAILABLE:
            logger.info("[bold cyan]4. Generating Evidently Reports[/]")
            _generate_evidently_report(train_features, inference_features, output_path)
        else:
            logger.info("[bold cyan]4. Evidently Reports[/] — Skipped (not installed)")

        # ── 5. Save Monitoring Results ───────────────────────────────────
        _save_monitoring_results(results, output_path)

        # ── 6. Alert Summary ─────────────────────────────────────────────
        if results["alerts"]:
            logger.info(f"[bold yellow]⚠ {len(results['alerts'])} alerts generated:[/]")
            for alert in results["alerts"]:
                severity_color = {"P1": "red", "P2": "yellow", "P3": "cyan", "P4": "dim"}.get(
                    alert["severity"], "white"
                )
                logger.info(
                    f"  [{severity_color}][{alert['severity']}][/] "
                    f"{alert['type']}: {alert['message']}"
                )
        else:
            logger.info("[bold green]✅ No alerts — all monitors within thresholds[/]")

        return results


def _compute_feature_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    psi_threshold: float = 0.2,
    ks_threshold: float = 0.05,
) -> dict:
    """
    Compute drift metrics for each feature.

    Uses:
      - PSI (Population Stability Index) for overall drift magnitude
      - KS test (Kolmogorov-Smirnov) for statistical significance
    """
    drift_results = {}
    common_cols = [c for c in reference.columns if c in current.columns]

    for col in common_cols:
        ref_vals = reference[col].dropna().values
        cur_vals = current[col].dropna().values

        if len(ref_vals) == 0 or len(cur_vals) == 0:
            continue

        # Ensure numeric
        if not np.issubdtype(ref_vals.dtype, np.number):
            continue

        result = {"feature": col}

        # PSI
        try:
            psi = _calculate_psi(ref_vals, cur_vals, n_bins=10)
            result["psi"] = float(psi)
            result["psi_drifted"] = psi > psi_threshold
        except Exception:
            result["psi"] = None
            result["psi_drifted"] = False

        # KS Test
        try:
            ks_stat, ks_pvalue = stats.ks_2samp(ref_vals, cur_vals)
            result["ks_statistic"] = float(ks_stat)
            result["ks_pvalue"] = float(ks_pvalue)
            result["ks_drifted"] = ks_pvalue < ks_threshold
        except Exception:
            result["ks_statistic"] = None
            result["ks_pvalue"] = None
            result["ks_drifted"] = False

        # Overall drift flag
        result["is_drifted"] = result.get("psi_drifted", False) or result.get("ks_drifted", False)

        # Distribution stats
        result["ref_mean"] = float(np.mean(ref_vals))
        result["cur_mean"] = float(np.mean(cur_vals))
        result["mean_shift"] = float(
            abs(np.mean(cur_vals) - np.mean(ref_vals)) / (np.std(ref_vals) + 1e-8)
        )

        drift_results[col] = result

    return drift_results


def _calculate_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """
    Calculate Population Stability Index (PSI).

    PSI < 0.1: No significant shift
    0.1 ≤ PSI < 0.2: Moderate shift, investigate
    PSI ≥ 0.2: Significant shift, action needed
    """
    # Create bins from reference distribution
    bins = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    bins[0] = -np.inf
    bins[-1] = np.inf
    bins = np.unique(bins)

    # Calculate proportions
    ref_counts = np.histogram(reference, bins=bins)[0]
    cur_counts = np.histogram(current, bins=bins)[0]

    # Normalize to proportions (with smoothing)
    ref_pct = (ref_counts + 1) / (len(reference) + len(bins) - 1)
    cur_pct = (cur_counts + 1) / (len(current) + len(bins) - 1)

    # PSI formula
    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return psi


def _analyze_prediction_distribution(preds_df: pd.DataFrame, model_id: str) -> dict:
    """Analyze the distribution of model predictions."""
    result = {"model_id": model_id, "total_records": len(preds_df)}

    if "prediction" in preds_df.columns:
        preds = preds_df["prediction"].dropna()
        result["pred_mean"] = float(preds.mean())
        result["pred_std"] = float(preds.std())
        result["pred_min"] = float(preds.min())
        result["pred_max"] = float(preds.max())
        result["pred_median"] = float(preds.median())

        # Check for anomalous distributions
        if preds.std() < 1e-10:
            result["warning"] = "constant_predictions"
        elif preds.nunique() == 1:
            result["warning"] = "single_value"

    if "probability" in preds_df.columns:
        probs = preds_df["probability"].dropna()
        result["prob_mean"] = float(probs.mean())
        result["prob_p10"] = float(probs.quantile(0.1))
        result["prob_p50"] = float(probs.quantile(0.5))
        result["prob_p90"] = float(probs.quantile(0.9))

    if "cluster_id" in preds_df.columns:
        result["n_clusters"] = int(preds_df["cluster_id"].nunique())
        result["cluster_sizes"] = preds_df["cluster_id"].value_counts().to_dict()

    if "error" in preds_df.columns:
        result["error_count"] = int(preds_df["error"].notna().sum())
        result["error_rate"] = float(preds_df["error"].notna().mean())

    logger.info(
        f"  {model_id}: "
        f"{result.get('total_records', 0):,} records, "
        f"mean={result.get('pred_mean', result.get('prob_mean', 'N/A'))}"
    )

    return result


def _compute_data_quality(df: pd.DataFrame, null_threshold: float = 0.1) -> dict:
    """Compute data quality metrics for each feature."""
    quality = {}

    for col in df.columns:
        result = {
            "feature": col,
            "null_rate": float(df[col].isnull().mean()),
            "unique_count": int(df[col].nunique()),
            "unique_rate": float(df[col].nunique() / len(df)),
        }

        if np.issubdtype(df[col].dtype, np.number):
            result["mean"] = float(df[col].mean())
            result["std"] = float(df[col].std())
            result["min"] = float(df[col].min())
            result["max"] = float(df[col].max())
            result["zero_rate"] = float((df[col] == 0).mean())
            result["inf_count"] = int(np.isinf(df[col].fillna(0)).sum())

        # Flag issues
        issues = []
        if result["null_rate"] > null_threshold:
            issues.append(f"high_null_rate ({result['null_rate']:.2%})")
        if result["unique_count"] <= 1:
            issues.append("constant_column")
        if result.get("inf_count", 0) > 0:
            issues.append(f"contains_inf ({result['inf_count']})")

        result["has_issue"] = len(issues) > 0
        result["issues"] = issues

        quality[col] = result

    issues_count = sum(1 for v in quality.values() if v["has_issue"])
    logger.info(f"  Data quality: {len(quality)} features checked, {issues_count} with issues")

    return quality


def _generate_evidently_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_path: str,
) -> None:
    """Generate Evidently AI drift and quality reports."""
    try:
        # Sample for performance (Evidently can be slow on large datasets)
        sample_size = min(5000, len(reference), len(current))
        ref_sample = reference.sample(n=sample_size, random_state=42)
        cur_sample = current.sample(n=sample_size, random_state=42)

        # Select only numeric columns
        numeric_cols = ref_sample.select_dtypes(include=[np.number]).columns
        common_cols = [c for c in numeric_cols if c in cur_sample.columns]

        # Limit columns for report performance
        report_cols = common_cols[:30]
        ref_report = ref_sample[report_cols]
        cur_report = cur_sample[report_cols]

        # Data Drift Report
        drift_report = Report(metrics=[DataDriftPreset()])
        drift_report.run(
            reference_data=ref_report,
            current_data=cur_report,
        )

        report_path = Path(output_path)
        report_path.mkdir(parents=True, exist_ok=True)

        drift_html = report_path / "data_drift_report.html"
        drift_report.save_html(str(drift_html))
        logger.info(f"  Evidently drift report → [cyan]{drift_html}[/]")

        # Data Quality Report
        quality_report = Report(metrics=[DataQualityPreset()])
        quality_report.run(
            reference_data=ref_report,
            current_data=cur_report,
        )

        quality_html = report_path / "data_quality_report.html"
        quality_report.save_html(str(quality_html))
        logger.info(f"  Evidently quality report → [cyan]{quality_html}[/]")

    except Exception as e:
        logger.warning(f"  Evidently report generation failed: {e}")


def _save_monitoring_results(results: dict, output_path: str) -> None:
    """Save monitoring results as JSON and Parquet."""
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save feature drift as Parquet
    if results["feature_drift"]:
        drift_rows = list(results["feature_drift"].values())
        drift_df = pd.DataFrame(drift_rows)
        save_parquet(
            drift_df,
            f"{output_path}feature_drift_metrics.parquet",
            "Feature drift metrics",
        )

    # Save prediction stats as Parquet
    if results["prediction_drift"]:
        pred_rows = list(results["prediction_drift"].values())
        pred_df = pd.DataFrame(pred_rows)
        save_parquet(
            pred_df,
            f"{output_path}prediction_stats.parquet",
            "Prediction statistics",
        )

    # Save alerts as JSON
    alerts_path = output_dir / "alerts.json"
    with open(alerts_path, "w") as f:
        json.dump(results["alerts"], f, indent=2, default=str)
    logger.info(f"  Alerts → [cyan]{alerts_path}[/]")
