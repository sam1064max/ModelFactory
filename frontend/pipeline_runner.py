"""
Streamlit-Aware Pipeline Runner
───────────────────────────────────────────────────────────────────────────────
Wraps the existing backend pipeline stages for Streamlit consumption.

Key differences from orchestration.pipeline_runner:
  1. Each stage is callable independently with progress callbacks
  2. Log output is captured into a deque for real-time display
  3. Stage status is communicated via a shared dict (not console prints)
  4. Results are returned as structured dicts for st.metric() consumption

This module does NOT duplicate business logic — it delegates to src/* modules.
"""

from __future__ import annotations

import logging
import sys
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Ensure project root is on the path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import ray

from src.batch_inference import run_batch_inference
from src.data_ingestion import generate_synthetic_data
from src.feature_engineering import run_feature_engineering
from src.model_monitoring import run_monitoring
from src.model_training import train_all_models
from src.utils import load_config, load_model_registry

# ── Stage Definitions ─────────────────────────────────────────────────────────

STAGES = [
    "data_ingestion",
    "feature_engineering",
    "model_training",
    "batch_inference",
    "monitoring",
]

STAGE_LABELS = {
    "data_ingestion": "Data Ingestion",
    "feature_engineering": "Feature Engineering (Spark)",
    "model_training": "Parallel Training (Ray)",
    "batch_inference": "Batch Inference",
    "monitoring": "Monitoring & Drift Detection",
}


# ── Log Capture ───────────────────────────────────────────────────────────────


class StreamlitLogHandler(logging.Handler):
    """Captures log records into a shared deque for Streamlit display."""

    def __init__(self, log_queue: deque, maxlen: int = 500):
        super().__init__()
        self.log_queue = log_queue
        # Configure a verbose, detailed format
        self.setFormatter(
            logging.Formatter(
                "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
                datefmt="%H:%M:%S",
            )
        )

    def emit(self, record: logging.Record) -> None:
        import re

        msg = str(record.msg)
        if record.args:
            try:
                msg = msg % record.args
            except Exception:
                pass
        clean = re.sub(r"\[/?\w+(?:=#[0-9a-fA-F]+)?\]", "", msg)
        clean = re.sub(r"\[/?\w+\]", "", clean)

        # Temporarily override record fields to format the clean msg
        orig_msg = record.msg
        orig_args = record.args
        record.msg = clean
        record.args = ()

        formatted = self.format(record)

        # Restore original fields
        record.msg = orig_msg
        record.args = orig_args

        self.log_queue.append(formatted)


# ── Stage Status ──────────────────────────────────────────────────────────────

STAGE_IDLE = "idle"
STAGE_RUNNING = "running"
STAGE_DONE = "done"
STAGE_FAILED = "failed"


# ── Pipeline Runner ───────────────────────────────────────────────────────────


class StreamlitPipelineRunner:
    """
    Runs the full MLOps pipeline with progress reporting for Streamlit.

    Usage:
        runner = StreamlitPipelineRunner()
        results = runner.run(
            on_log=lambda msg: ...,
            on_stage=lambda name, status, data: ...,
        )
    """

    def __init__(self, config: dict | None = None, model_registry: dict | None = None):
        self.config = config or load_config()
        self.model_registry = model_registry or load_model_registry()

        # Override training_rows if set via Streamlit sidebar
        self._override_training_rows: int | None = None
        self._override_inference_rows: int | None = None
        self._override_num_models: int | None = None

        self._log_queue: deque = deque(maxlen=500)
        self._setup_log_capture()

    # ── Configuration Overrides ───────────────────────────────────────────

    def set_training_rows(self, n: int) -> None:
        self._override_training_rows = n

    def set_inference_rows(self, n: int) -> None:
        self._override_inference_rows = n

    def set_num_models(self, n: int) -> None:
        self._override_num_models = n

    def _apply_overrides(self) -> None:
        if self._override_training_rows:
            self.config["data"]["synthetic"]["training_rows"] = self._override_training_rows
        if self._override_inference_rows:
            self.config["data"]["synthetic"]["inference_rows"] = self._override_inference_rows
        if self._override_num_models is not None:
            models = self.model_registry.get("models", [])
            self.model_registry["models"] = models[: self._override_num_models]

    # ── Log Capture ──────────────────────────────────────────────────────

    def _setup_log_capture(self) -> None:
        """Attach a StreamlitLogHandler to the root pipeline logger."""
        handler = StreamlitLogHandler(self._log_queue)
        # Use the existing logger from src.utils
        logger = logging.getLogger("mlops_pipeline")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        # Also capture Ray logs
        ray_logger = logging.getLogger("ray")
        ray_logger.addHandler(handler)
        ray_logger.setLevel(logging.WARNING)

    def get_logs(self, n: int = 50) -> list[str]:
        """Return the most recent N log lines."""
        return list(self._log_queue)[-n:]

    # ── Pipeline Execution ───────────────────────────────────────────────

    def run(
        self,
        on_log: Callable[[str], None] | None = None,
        on_stage: Callable[[str, str, dict], None] | None = None,
        on_progress: Callable[[float], None] | None = None,
    ) -> dict[str, Any]:
        """
        Execute the full pipeline.

        Args:
            on_log: Called with each log line as it's produced.
            on_stage: Called when a stage changes status.
                      Args: (stage_name, status, data_dict)
            on_progress: Called with overall progress fraction (0.0–1.0).

        Returns:
            dict with keys: training_results, predictions, monitoring_results,
                            total_time, stage_times, summary_metrics
        """
        self._apply_overrides()
        stage_times: dict[str, float] = {}
        pipeline_start = time.time()

        # ── Init Ray ─────────────────────────────────────────────────────
        self._emit_log("Initializing Ray...", on_log)
        ray_config = self.config.get("ray", {})
        num_cpus = ray_config.get("num_cpus", None)
        if not ray.is_initialized():
            ray.init(num_cpus=num_cpus, log_to_driver=False, ignore_reinit_error=True)
        ray_info = ray.cluster_resources()
        self._emit_log(
            f"Ray initialized: {ray_info.get('CPU', 0):.0f} CPUs",
            on_log,
        )

        try:
            # ── Stage 1: Data Ingestion ──────────────────────────────────
            self._update_stage("data_ingestion", STAGE_RUNNING, {}, on_stage)
            self._emit_log("Generating synthetic training data...", on_log)
            t0 = time.time()
            train_df, inference_df = generate_synthetic_data(self.config)
            elapsed = time.time() - t0
            stage_times["data_ingestion"] = elapsed
            self._emit_log(
                f"Training: {len(train_df):,} rows, Inference: {len(inference_df):,} rows",
                on_log,
            )
            self._update_stage(
                "data_ingestion",
                STAGE_DONE,
                {"rows": len(train_df), "inference_rows": len(inference_df)},
                on_stage,
            )
            self._update_progress(0.2, on_progress)

            # ── Stage 2: Feature Engineering ─────────────────────────────
            self._update_stage("feature_engineering", STAGE_RUNNING, {}, on_stage)
            self._emit_log("Running feature transformations...", on_log)
            t0 = time.time()
            train_features, inference_features, feature_pipeline = run_feature_engineering(
                train_df, inference_df, self.config
            )
            elapsed = time.time() - t0
            stage_times["feature_engineering"] = elapsed
            self._emit_log(
                f"Features: {train_features.shape[1]} engineered features",
                on_log,
            )
            self._update_stage(
                "feature_engineering",
                STAGE_DONE,
                {"num_features": train_features.shape[1]},
                on_stage,
            )
            self._update_progress(0.4, on_progress)

            # ── Stage 3: Model Training ──────────────────────────────────
            self._update_stage("model_training", STAGE_RUNNING, {}, on_stage)
            self._emit_log("Launching Ray Tune HPO...", on_log)
            t0 = time.time()
            training_results = train_all_models(
                train_features=train_features,
                train_df=train_df,
                feature_pipeline=feature_pipeline,
                model_registry=self.model_registry,
                config=self.config,
            )
            elapsed = time.time() - t0
            stage_times["model_training"] = elapsed
            successful = sum(1 for r in training_results.values() if r.get("status") == "success")
            self._emit_log(
                f"Training complete: {successful}/{len(training_results)} models",
                on_log,
            )
            self._update_stage(
                "model_training",
                STAGE_DONE,
                {"trained": successful, "total": len(training_results)},
                on_stage,
            )
            self._update_progress(0.6, on_progress)

            # ── Stage 4: Batch Inference ─────────────────────────────────
            self._update_stage("batch_inference", STAGE_RUNNING, {}, on_stage)
            self._emit_log("Running batch inference via Ray Actors...", on_log)
            t0 = time.time()
            predictions = run_batch_inference(
                inference_features=inference_features,
                feature_pipeline=feature_pipeline,
                training_results=training_results,
                model_registry=self.model_registry,
                config=self.config,
            )
            elapsed = time.time() - t0
            stage_times["batch_inference"] = elapsed
            total_preds = sum(len(p) for p in predictions.values())
            self._emit_log(f"Inference complete: {total_preds:,} predictions", on_log)
            self._update_stage(
                "batch_inference", STAGE_DONE, {"predictions": total_preds}, on_stage
            )
            self._update_progress(0.8, on_progress)

            # ── Stage 5: Monitoring ──────────────────────────────────────
            self._update_stage("monitoring", STAGE_RUNNING, {}, on_stage)
            self._emit_log("Computing drift detection...", on_log)
            t0 = time.time()
            monitoring_results = run_monitoring(
                train_features=train_features,
                inference_features=inference_features,
                predictions=predictions,
                config=self.config,
            )
            elapsed = time.time() - t0
            stage_times["monitoring"] = elapsed
            alerts = monitoring_results.get("alerts", [])
            self._emit_log(
                f"Monitoring complete: {len(alerts)} alerts generated",
                on_log,
            )
            self._update_stage("monitoring", STAGE_DONE, {"alerts": len(alerts)}, on_stage)
            self._update_progress(1.0, on_progress)

        finally:
            if ray.is_initialized():
                ray.shutdown()
                self._emit_log("Ray shutdown complete", on_log)

        total_time = time.time() - pipeline_start

        # ── Build Summary Metrics ────────────────────────────────────────
        successful_train = sum(1 for r in training_results.values() if r.get("status") == "success")
        failed_train = len(training_results) - successful_train
        total_preds = sum(len(p) for p in predictions.values())
        drift = monitoring_results.get("feature_drift", {})
        drifted = sum(1 for v in drift.values() if v.get("is_drifted", False))

        # Compute average primary metric
        metric_values = []
        for r in training_results.values():
            metrics = r.get("best_metrics", {})
            if "roc_auc" in metrics:
                metric_values.append(metrics["roc_auc"])
            elif "r2" in metrics:
                metric_values.append(metrics["r2"])
            elif "silhouette_score" in metrics:
                metric_values.append(metrics["silhouette_score"])
        avg_metric = sum(metric_values) / len(metric_values) if metric_values else 0.0

        summary_metrics = {
            "models_trained": successful_train,
            "models_failed": failed_train,
            "num_features": train_features.shape[1],
            "training_time": stage_times.get("model_training", 0),
            "inference_records": total_preds,
            "avg_accuracy": avg_metric,
            "total_time": total_time,
            "features_drifted": drifted,
            "total_features_monitored": len(drift),
            "alerts": len(monitoring_results.get("alerts", [])),
        }

        self._emit_log(
            f"Pipeline complete in {total_time:.1f}s — "
            f"{successful_train} models, {total_preds:,} predictions",
            on_log,
        )

        return {
            "training_results": training_results,
            "predictions": predictions,
            "monitoring_results": monitoring_results,
            "total_time": total_time,
            "stage_times": stage_times,
            "summary_metrics": summary_metrics,
        }

    # ── Internal Helpers ─────────────────────────────────────────────────

    @staticmethod
    def _emit_log(msg: str, callback: Callable | None) -> None:
        if callback:
            callback(msg)

    @staticmethod
    def _update_stage(
        name: str,
        status: str,
        data: dict,
        callback: Callable | None,
    ) -> None:
        if callback:
            callback(name, status, data)

    @staticmethod
    def _update_progress(value: float, callback: Callable | None) -> None:
        if callback:
            callback(value)
