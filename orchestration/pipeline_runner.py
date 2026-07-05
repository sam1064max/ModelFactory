"""
Pipeline Runner — Ray-Integrated Local Orchestration
──────────────────────────────────────────────────────────────────────────────
Orchestrates the full MLOps pipeline end-to-end with Ray for training and
inference stages.

Ray lifecycle:
  - ray.init() at pipeline start (local mode: uses all available CPU cores)
  - ray.shutdown() at pipeline end (releases resources)

In production (Databricks), this logic is replaced by:
  - Databricks Workflows (multi-task DAGs with dependencies)
  - ray.util.spark.setup_ray_cluster() for Ray on Databricks
  - Databricks Asset Bundles for deployment
  - Scheduled triggers (cron) or event-driven triggers (data arrival)

Usage:
    python -m orchestration.pipeline_runner
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import ray

from src.batch_inference import run_batch_inference
from src.data_ingestion import generate_synthetic_data
from src.feature_engineering import run_feature_engineering
from src.model_monitoring import run_monitoring
from src.model_training import train_all_models
from src.utils import (
    load_config,
    load_model_registry,
    logger,
    timer,
    console,
)


def main():
    """Execute the full MLOps pipeline with Ray."""
    console.rule("[bold magenta]MLOps Platform — Ray Pipeline Runner[/]")
    pipeline_start = time.time()

    # ── Load Configuration ───────────────────────────────────────────────
    config = load_config()
    model_registry = load_model_registry()

    # ── Initialize Ray ───────────────────────────────────────────────────
    # In production (Databricks): ray.util.spark.setup_ray_cluster()
    # Locally: ray.init() uses all available CPU cores
    ray_config = config.get("ray", {})
    num_cpus = ray_config.get("num_cpus", None)  # None = auto-detect

    console.rule("[bold yellow]Initializing Ray[/]")
    if not ray.is_initialized():
        ray.init(
            num_cpus=num_cpus,
            log_to_driver=False,
            ignore_reinit_error=True,
        )
    ray_info = ray.cluster_resources()
    logger.info(
        f"Ray initialized: {ray_info.get('CPU', 0):.0f} CPUs, "
        f"{ray_info.get('GPU', 0):.0f} GPUs, "
        f"{ray_info.get('memory', 0) / 1e9:.1f} GB memory"
    )

    try:
        # ── Stage 1: Data Ingestion (No Ray — Spark in production) ───────
        console.rule("[bold cyan]Stage 1: Data Ingestion[/]")
        train_df, inference_df = generate_synthetic_data(config)

        # ── Stage 2: Feature Engineering (No Ray — Spark in production) ──
        console.rule("[bold cyan]Stage 2: Feature Engineering[/]")
        train_features, inference_features, feature_pipeline = run_feature_engineering(
            train_df, inference_df, config
        )

        # ── Stage 3: Model Training (Ray Tune + Ray Train) ──────────────
        console.rule("[bold cyan]Stage 3: Model Training (Ray Tune)[/]")
        training_results = train_all_models(
            train_features=train_features,
            train_df=train_df,
            feature_pipeline=feature_pipeline,
            model_registry=model_registry,
            config=config,
        )

        # ── Stage 4: Batch Inference (Ray Actors) ────────────────────────
        console.rule("[bold cyan]Stage 4: Batch Inference (Ray Actors)[/]")
        predictions = run_batch_inference(
            inference_features=inference_features,
            feature_pipeline=feature_pipeline,
            training_results=training_results,
            model_registry=model_registry,
            config=config,
        )

        # ── Stage 5: Monitoring (No Ray — Spark SQL in production) ───────
        console.rule("[bold cyan]Stage 5: Monitoring & Observability[/]")
        monitoring_results = run_monitoring(
            train_features=train_features,
            inference_features=inference_features,
            predictions=predictions,
            config=config,
        )

        # ── Pipeline Summary ─────────────────────────────────────────────
        total_time = time.time() - pipeline_start
        console.rule("[bold green]Pipeline Complete[/]")

        _print_summary(
            training_results=training_results,
            predictions=predictions,
            monitoring_results=monitoring_results,
            total_time=total_time,
        )

    finally:
        # ── Shutdown Ray ─────────────────────────────────────────────────
        if ray.is_initialized():
            ray.shutdown()
            logger.info("Ray shutdown complete")


def _print_summary(
    training_results: dict,
    predictions: dict,
    monitoring_results: dict,
    total_time: float,
) -> None:
    """Print a formatted pipeline summary."""
    from rich.table import Table

    # Training summary
    table = Table(title="Pipeline Execution Summary (Ray)", show_lines=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value", style="green")

    # Training stats
    successful = sum(1 for r in training_results.values() if r.get("status") == "success")
    failed = len(training_results) - successful
    table.add_row("Models Trained", f"{successful} successful, {failed} failed")
    table.add_row("Training Engine", "Ray Tune (ASHA scheduler)")

    # Inference stats
    total_preds = sum(len(p) for p in predictions.values())
    table.add_row("Total Predictions", f"{total_preds:,}")
    table.add_row("Models Scored", str(len(predictions)))
    table.add_row("Inference Engine", "Ray Actors (stateful)")

    # Monitoring stats
    alerts = monitoring_results.get("alerts", [])
    p1_alerts = sum(1 for a in alerts if a["severity"] == "P1")
    p2_alerts = sum(1 for a in alerts if a["severity"] == "P2")
    table.add_row(
        "Alerts",
        f"{len(alerts)} total ({p1_alerts} P1, {p2_alerts} P2)",
    )

    # Drift stats
    drift = monitoring_results.get("feature_drift", {})
    drifted = sum(1 for v in drift.values() if v.get("is_drifted", False))
    table.add_row(
        "Feature Drift",
        f"{drifted}/{len(drift)} features drifted",
    )

    # Timing
    if total_time < 60:
        time_str = f"{total_time:.1f} seconds"
    else:
        time_str = f"{total_time / 60:.1f} minutes"
    table.add_row("Total Pipeline Time", time_str)

    # MLflow info
    table.add_row(
        "MLflow UI",
        "Run: mlflow ui --port 5000",
    )

    console.print(table)

    # Print model-level results
    model_table = Table(title="Model Results", show_lines=True)
    model_table.add_column("Model ID", style="cyan")
    model_table.add_column("Type")
    model_table.add_column("Status")
    model_table.add_column("Primary Metric")
    model_table.add_column("Decision")

    for model_id, result in training_results.items():
        status = result.get("status", "unknown")
        status_str = "[green]✅ Success[/]" if status == "success" else "[red]❌ Failed[/]"
        model_type = result.get("model_type", "unknown")
        metrics = result.get("best_metrics", {})

        # Get primary metric based on model type
        if "roc_auc" in metrics:
            primary = f"AUC={metrics['roc_auc']:.4f}"
        elif "r2" in metrics:
            primary = f"R²={metrics['r2']:.4f}"
        elif "silhouette_score" in metrics:
            primary = f"Sil={metrics['silhouette_score']:.4f}"
        else:
            primary = "N/A"

        decision = result.get("promotion_decision", "N/A")

        model_table.add_row(model_id, model_type, status_str, primary, decision)

    console.print(model_table)


if __name__ == "__main__":
    main()
