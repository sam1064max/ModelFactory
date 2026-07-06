"""
Batch Inference Module — Ray Actor Integration
──────────────────────────────────────────────────────────────────────────────
Implements stateful batch inference using Ray Actors:
  1. Each Ray Actor loads one model and keeps it in memory
  2. Data partitions are streamed to actors via the Ray object store
  3. When an actor finishes its model, it swaps to the next (warm swap)
  4. Results are collected and written as Parquet

In production (Databricks), this uses:
  - Ray Actors on Databricks clusters for stateful model scoring
  - Spark for Delta Lake I/O (read universe, write results)
  - Apache Arrow for zero-copy Spark ↔ Ray data transfer
  - Delta Lake partition-overwrite for idempotent writes
"""

import time
from typing import Any

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import ray

from src.feature_engineering import FeatureEngineeringPipeline
from src.utils import (
    get_model_type_category,
    logger,
    save_parquet,
    timer,
)

# ── Ray Actor: Stateful Model Scorer ─────────────────────────────────────────


@ray.remote
class ModelScoringActor:
    """
    Stateful Ray Actor that holds a model in memory for scoring.

    Key advantages over loading-per-chunk:
    - Model loaded once, stays in memory across all data batches
    - No repeated deserialisation (10-500MB per model in production)
    - Automatic restart on failure (max_restarts configurable)
    - Warm swap: actor can switch to a new model without recreation

    In production, actors run on Ray workers co-located with Spark executors
    on the same Databricks cluster nodes.
    """

    def __init__(self, actor_id: int):
        self.actor_id = actor_id
        self.model: Any = None
        self.model_id: str | None = None
        self.model_type: str | None = None
        self.selected_features: list[str] | None = None
        self.records_scored = 0
        self.models_completed = 0

    def load_model(
        self,
        model_id: str,
        run_id: str,
        model_type: str,
        selected_features: list[str],
    ) -> str:
        """Load a model from MLflow into this actor's memory."""
        # Release previous model if any (warm swap)
        if self.model is not None:
            del self.model
            self.models_completed += 1

        model_uri = f"runs:/{run_id}/model"
        logger.info(
            f"  [Actor {self.actor_id}] Loading model '{model_id}' from MLflow registry ({model_uri})..."
        )
        self.model = mlflow.pyfunc.load_model(model_uri)
        self.model_id = model_id
        self.model_type = model_type
        self.selected_features = selected_features
        self.records_scored = 0
        logger.info(
            f"  [Actor {self.actor_id}] Successfully loaded model '{model_id}' into worker memory."
        )
        return f"Actor {self.actor_id}: loaded {model_id}"

    def score_batch(
        self,
        data_batch: pd.DataFrame,
        include_probs: bool = True,
    ) -> pd.DataFrame:
        """
        Score a batch of data against the model held in memory.

        The model is NOT loaded per batch — it persists in the actor.
        Only data moves; the model stays put.
        """
        assert self.model_type is not None, "Model type not loaded"
        category = get_model_type_category(self.model_type)

        # Column pruning: only select features this model needs
        assert self.selected_features is not None, "Selected features not loaded"
        available_features = [f for f in self.selected_features if f in data_batch.columns]
        if not available_features:
            raise ValueError(
                f"No matching features for {self.model_id}. "
                f"Expected: {self.selected_features[:5]}..."
            )

        chunk = data_batch[available_features].fillna(0)

        result = pd.DataFrame()
        result["model_id"] = self.model_id
        result["record_index"] = range(len(chunk))

        assert self.model is not None, "Model not loaded"
        logger.info(
            f"  [Actor {self.actor_id}] Scoring batch chunk of size {len(chunk):,} for model '{self.model_id}'..."
        )
        try:
            raw_predictions = self.model.predict(chunk)

            if category == "classifier":
                result["prediction"] = raw_predictions
                if include_probs:
                    try:
                        unwrapped = self.model._model_impl
                        if hasattr(unwrapped, "predict_proba"):
                            probs = unwrapped.predict_proba(chunk)
                            if probs.ndim == 2 and probs.shape[1] >= 2:
                                result["probability"] = probs[:, 1]
                            else:
                                result["probability"] = probs.ravel()
                        else:
                            result["probability"] = raw_predictions.astype(float)
                    except Exception:
                        result["probability"] = raw_predictions.astype(float)

            elif category == "regressor":
                result["prediction"] = raw_predictions

            elif category == "clusterer":
                result["cluster_id"] = raw_predictions

        except Exception as e:
            result["prediction"] = np.nan
            result["error"] = str(e)

        self.records_scored += len(result)
        return result

    def get_stats(self) -> dict:
        """Return actor statistics."""
        return {
            "actor_id": self.actor_id,
            "current_model": self.model_id,
            "records_scored": self.records_scored,
            "models_completed": self.models_completed,
        }


# ── Main Inference Entry Point ───────────────────────────────────────────────


def run_batch_inference(
    inference_features: pd.DataFrame,
    feature_pipeline: FeatureEngineeringPipeline,
    training_results: dict[str, dict],
    model_registry: dict,
    config: dict,
) -> dict[str, pd.DataFrame]:
    """
    Run batch inference using a pool of Ray Actors.

    Architecture:
    1. Create an actor pool (N actors, each holding one model)
    2. Put the inference data into the Ray object store (once)
    3. Each actor loads its model and scores all data chunks
    4. When an actor finishes, it swaps to the next unscored model
    5. Results are collected and saved as Parquet

    This is the "Read Once, Score Many" pattern:
    - Data is placed in the Ray object store once
    - Multiple actors read the same data (zero-copy on same node)
    - Models stay in actor memory — no broadcast, no deserialisation

    Args:
        inference_features: Pre-computed inference features
        feature_pipeline: Fitted feature engineering pipeline
        training_results: Results from training (includes selected features)
        model_registry: Model registry config
        config: Pipeline configuration

    Returns:
        dict: Mapping of model_id → predictions DataFrame
    """
    inference_config = config["inference"]
    chunk_size = inference_config.get("chunk_size", 10000)
    include_probs = inference_config.get("include_probabilities", True)
    output_path = config["data"]["paths"]["inference_output"]

    # Get successfully trained models
    successful_models = {
        model_id: result
        for model_id, result in training_results.items()
        if result.get("status") == "success"
    }

    if not successful_models:
        logger.warning("No successfully trained models — skipping inference.")
        return {}

    # Determine actor pool size (min of available models and config)
    max_actors = inference_config.get("max_parallel_models", 5)
    num_actors = min(max_actors, len(successful_models))

    logger.info(
        f"[bold]Starting Ray Actor inference:[/] "
        f"{len(successful_models)} models × "
        f"{len(inference_features):,} records "
        f"(actor pool: {num_actors})"
    )

    # Put inference data into Ray object store once
    # In production: Spark reads from Delta Lake → Arrow → Ray object store
    data_chunks = _partition_data(inference_features, chunk_size)
    chunk_refs = [ray.put(chunk) for chunk in data_chunks]

    logger.info(
        f"  Data partitioned into {len(chunk_refs)} chunks "
        f"({chunk_size:,} records each), stored in Ray object store"
    )

    all_predictions = {}
    total_predictions = 0
    total_time = 0.0

    with timer(f"Batch Inference (Ray Actors) — {len(successful_models)} Models"):
        # Create actor pool
        actors = [ModelScoringActor.remote(i) for i in range(num_actors)]  # type: ignore[attr-defined]

        # Build work queue: list of (model_id, result) pairs
        model_queue = list(successful_models.items())
        model_idx = 0

        # Assign initial models to actors
        active_tasks = {}  # {future: (actor, model_id)}
        for actor in actors:
            if model_idx < len(model_queue):
                model_id, result = model_queue[model_idx]
                model_idx += 1

                # Load model into actor
                ray.get(
                    actor.load_model.remote(
                        model_id=model_id,
                        run_id=result["run_id"],
                        model_type=result["model_type"],
                        selected_features=result["selected_features"],
                    )
                )

                # Launch scoring for all chunks
                future = _score_all_chunks.remote(actor, chunk_refs, include_probs)
                active_tasks[future] = (actor, model_id)

        # Process completions and reassign actors
        completed = 0
        while active_tasks:
            done, _ = ray.wait(list(active_tasks.keys()), num_returns=1)

            for future in done:
                actor, model_id = active_tasks.pop(future)
                completed += 1
                model_start = time.time()

                try:
                    predictions = ray.get(future)
                    elapsed = time.time() - model_start

                    # Save results
                    output_file = f"{output_path}{model_id}_predictions.parquet"
                    save_parquet(
                        predictions,
                        output_file,
                        f"Predictions ({model_id})",
                    )

                    all_predictions[model_id] = predictions
                    num_preds = len(predictions)
                    total_predictions += num_preds
                    total_time += elapsed

                    throughput = num_preds / elapsed if elapsed > 0 else 0
                    logger.info(
                        f"  [{completed}/{len(successful_models)}] "
                        f"[green]{model_id}[/]: {num_preds:,} predictions "
                        f"({throughput:,.0f} rec/sec)"
                    )

                except Exception as e:
                    logger.error(
                        f"  [{completed}/{len(successful_models)}] "
                        f"[red]{model_id}[/]: Inference failed — {e}"
                    )

                # Reassign actor to next model (warm swap)
                if model_idx < len(model_queue):
                    next_model_id, next_result = model_queue[model_idx]
                    model_idx += 1

                    # Swap model in the existing actor (no actor recreation)
                    ray.get(
                        actor.load_model.remote(
                            model_id=next_model_id,
                            run_id=next_result["run_id"],
                            model_type=next_result["model_type"],
                            selected_features=next_result["selected_features"],
                        )
                    )

                    future = _score_all_chunks.remote(actor, chunk_refs, include_probs)
                    active_tasks[future] = (actor, next_model_id)

        # Collect actor stats
        all_stats = ray.get([actor.get_stats.remote() for actor in actors])
        total_actor_records = sum(s["records_scored"] for s in all_stats)
        total_actor_models = sum(s["models_completed"] for s in all_stats)
        logger.info(
            f"  Actor pool stats: {total_actor_records:,} total records scored, "
            f"{total_actor_models} model swaps"
        )

    # ── Summary ──────────────────────────────────────────────────────────
    avg_throughput = total_predictions / total_time if total_time > 0 else 0
    logger.info(
        f"[bold]Inference complete:[/] "
        f"{total_predictions:,} total predictions, "
        f"{avg_throughput:,.0f} avg rec/sec"
    )

    # Generate inference summary
    _generate_inference_summary(all_predictions, output_path)

    return all_predictions


@ray.remote
def _score_all_chunks(
    actor: ModelScoringActor,
    chunk_refs: list,
    include_probs: bool,
) -> pd.DataFrame:
    """
    Score all data chunks using a single actor (the actor holds the model).

    Each chunk_ref is a reference in the Ray object store — zero-copy read
    on the same node, network transfer across nodes.
    """
    chunk_results = []
    for chunk_ref in chunk_refs:
        chunk_data = ray.get(chunk_ref)
        result = ray.get(actor.score_batch.remote(chunk_data, include_probs))  # type: ignore[attr-defined]
        chunk_results.append(result)

    return pd.concat(chunk_results, ignore_index=True)


def _partition_data(df: pd.DataFrame, chunk_size: int) -> list[pd.DataFrame]:
    """Split a DataFrame into chunks for parallel scoring."""
    n_rows = len(df)
    chunks = []
    for start in range(0, n_rows, chunk_size):
        end = min(start + chunk_size, n_rows)
        chunks.append(df.iloc[start:end].copy())
    return chunks


def _generate_inference_summary(
    all_predictions: dict[str, pd.DataFrame],
    output_path: str,
) -> None:
    """Generate a summary report of inference results."""
    summary_rows = []

    for model_id, preds in all_predictions.items():
        row = {
            "model_id": model_id,
            "total_records": len(preds),
        }

        if "prediction" in preds.columns:
            row["prediction_mean"] = preds["prediction"].mean()
            row["prediction_std"] = preds["prediction"].std()
            row["prediction_min"] = preds["prediction"].min()
            row["prediction_max"] = preds["prediction"].max()

        if "probability" in preds.columns:
            row["prob_mean"] = preds["probability"].mean()
            row["prob_std"] = preds["probability"].std()

        if "cluster_id" in preds.columns:
            row["n_clusters"] = preds["cluster_id"].nunique()
            row["cluster_distribution"] = str(preds["cluster_id"].value_counts().to_dict())

        if "error" in preds.columns:
            row["error_count"] = preds["error"].notna().sum()
        else:
            row["error_count"] = 0

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    save_parquet(
        summary_df,
        f"{output_path}inference_summary.parquet",
        "Inference summary",
    )
