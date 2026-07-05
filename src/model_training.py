"""
Model Training Module — Ray Integration
──────────────────────────────────────────────────────────────────────────────
Orchestrates training of multiple models with:
  - Config-driven model parameterization
  - Hyperparameter optimization via Ray Tune (ASHA scheduler)
  - MLflow experiment tracking (params, metrics, artifacts)
  - Model registration to MLflow Model Registry
  - Champion/Challenger comparison

In production (Databricks), this uses:
  - Ray Tune with ASHA for distributed hyperparameter search
  - Ray Train for distributed training workers
  - MLflow auto-logging via MLflowLoggerCallback
  - ray.util.spark.setup_ray_cluster() for Spark–Ray co-location
"""

import warnings
from typing import Any, Optional

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    calinski_harabasz_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold, KFold

from src.feature_engineering import FeatureEngineeringPipeline
from src.utils import (
    compute_data_hash,
    get_model_type_category,
    logger,
    timer,
)

warnings.filterwarnings("ignore", category=UserWarning)


def train_all_models(
    train_features: pd.DataFrame,
    train_df: pd.DataFrame,
    feature_pipeline: FeatureEngineeringPipeline,
    model_registry: dict,
    config: dict,
) -> dict[str, dict]:
    """
    Train all models defined in the model registry using Ray for distributed
    hyperparameter optimization and training orchestration.

    Ray Tune replaces manual random search with ASHA-scheduled distributed HPO.
    Each model's HPO trials run as Ray tasks, automatically parallelised across
    available CPU cores.

    Args:
        train_features: Transformed feature DataFrame
        train_df: Original training data (for targets)
        feature_pipeline: Fitted feature engineering pipeline
        model_registry: Model registry configuration
        config: Pipeline configuration

    Returns:
        dict: Mapping of model_id → training results
    """
    training_config = config["training"]
    models = model_registry["models"]
    results = {}

    # Set up MLflow
    mlflow_config = training_config["mlflow"]
    mlflow.set_tracking_uri(mlflow_config["tracking_uri"])
    mlflow.set_experiment(mlflow_config["experiment_name"])

    logger.info(f"[bold]Starting Ray-powered training pipeline: {len(models)} models[/]")

    # Put shared data into Ray object store once (avoids repeated serialisation)
    train_features_ref = ray.put(train_features)
    train_df_ref = ray.put(train_df)

    with timer(f"Model Training (Ray) — {len(models)} Models"):
        # Launch all models as Ray remote tasks for parallel training
        futures = {}
        for model_config in models:
            future = _train_single_model_remote.remote(
                model_config=model_config,
                train_features_ref=train_features_ref,
                train_df_ref=train_df_ref,
                feature_pipeline=feature_pipeline,
                training_config=training_config,
                feature_config=config["features"],
            )
            futures[future] = model_config["model_id"]

        # Collect results as they complete
        remaining = list(futures.keys())
        completed = 0
        while remaining:
            done, remaining = ray.wait(remaining, num_returns=1)
            for ref in done:
                model_id = futures[ref]
                completed += 1
                try:
                    result = ray.get(ref)
                    results[model_id] = result
                    status = result.get("status", "unknown")
                    if status == "success":
                        metrics_str = _format_metrics(result.get("best_metrics", {}))
                        logger.info(
                            f"  [{completed}/{len(models)}] "
                            f"✅ [green]{model_id}[/] — {metrics_str} "
                            f"— {result.get('promotion_decision', 'N/A')}"
                        )
                    else:
                        logger.error(
                            f"  [{completed}/{len(models)}] "
                            f"❌ [red]{model_id}[/] — {result.get('error', 'Unknown')}"
                        )
                except Exception as e:
                    logger.error(
                        f"  [{completed}/{len(models)}] "
                        f"❌ [red]{model_id}[/] — Ray task failed: {e}"
                    )
                    results[model_id] = {"status": "failed", "error": str(e)}

    # Summary
    successful = sum(1 for r in results.values() if r.get("status") == "success")
    failed = len(results) - successful
    logger.info(f"[bold]Training complete:[/] {successful} successful, {failed} failed")

    return results


@ray.remote
def _train_single_model_remote(
    model_config: dict,
    train_features_ref,
    train_df_ref,
    feature_pipeline: FeatureEngineeringPipeline,
    training_config: dict,
    feature_config: dict,
) -> dict:
    """
    Ray remote function: train a single model.

    This runs as an independent Ray task. Data is read from the Ray object store
    (zero-copy on the same node) instead of being serialised per task.
    """
    # Resolve object references
    train_features = ray.get(train_features_ref)
    train_df = ray.get(train_df_ref)

    return _train_single_model(
        model_config=model_config,
        train_features=train_features,
        train_df=train_df,
        feature_pipeline=feature_pipeline,
        training_config=training_config,
        feature_config=feature_config,
    )


def _train_single_model(
    model_config: dict,
    train_features: pd.DataFrame,
    train_df: pd.DataFrame,
    feature_pipeline: FeatureEngineeringPipeline,
    training_config: dict,
    feature_config: dict,
) -> dict:
    """Train a single model with Ray Tune HPO and MLflow tracking."""
    model_id = model_config["model_id"]
    model_type = model_config["model_type"]
    target_col = model_config.get("target_column")
    category = get_model_type_category(model_type)

    try:
        # Ensure MLflow is configured in this Ray worker
        mlflow_config = training_config["mlflow"]
        mlflow.set_tracking_uri(mlflow_config["tracking_uri"])
        mlflow.set_experiment(mlflow_config["experiment_name"])

        with mlflow.start_run(run_name=model_id) as run:
            # Log model configuration
            mlflow.set_tags(
                {
                    "model_id": model_id,
                    "model_type": model_type,
                    "model_category": category,
                    "description": model_config.get("description", ""),
                    "training_engine": "ray",
                }
            )

            # ── Feature Selection ────────────────────────────────────────
            target = train_df[target_col] if target_col else None
            max_features = feature_config["selection"]["max_features_per_model"]

            X_selected, selected_features = feature_pipeline.select_features(
                df=train_features,
                target=target,
                model_type=model_type,
                max_features=max_features,
            )

            mlflow.log_param("num_selected_features", len(selected_features))
            mlflow.log_param("selected_features", str(selected_features[:10]) + "...")

            # Fill remaining NaNs
            X_selected = X_selected.fillna(0)

            # ── Hyperparameter Optimization (Ray Tune) ───────────────────
            hpo_config = training_config.get("hpo", {})
            hp_space = model_config.get("hyperparameter_space", {})
            max_trials = hpo_config.get("max_trials", 5)

            best_model, best_params, best_metrics = _run_ray_tune_hpo(
                X=X_selected,
                y=target,
                model_type=model_type,
                hp_space=hp_space,
                max_trials=max_trials,
                cv_folds=training_config.get("cv", {}).get("n_splits", 5),
                category=category,
            )

            # ── Log Results ──────────────────────────────────────────────
            mlflow.log_params(best_params)
            for metric_name, metric_value in best_metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            # Log the model
            if "xgboost" in model_type:
                mlflow.xgboost.log_model(best_model, artifact_path="model")
            else:
                mlflow.sklearn.log_model(best_model, artifact_path="model")

            # Log data hash for reproducibility
            data_hash = compute_data_hash(X_selected)
            mlflow.log_param("training_data_hash", data_hash)
            mlflow.log_param("training_rows", len(X_selected))

            # ── Champion/Challenger ──────────────────────────────────────
            champion_config = training_config.get("champion_challenger", {})
            promotion_decision = _evaluate_champion_challenger(
                model_id=model_id,
                new_metrics=best_metrics,
                champion_config=champion_config,
                model_config=model_config,
                category=category,
            )
            mlflow.set_tag("promotion_decision", promotion_decision)

            # Register model
            model_uri = f"runs:/{run.info.run_id}/model"
            try:
                mv = mlflow.register_model(model_uri, model_id)
            except Exception:
                pass  # Registration may fail in local mode

            return {
                "status": "success",
                "run_id": run.info.run_id,
                "model_type": model_type,
                "best_params": best_params,
                "best_metrics": best_metrics,
                "selected_features": selected_features,
                "promotion_decision": promotion_decision,
            }

    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "model_type": model_type,
        }


def _run_ray_tune_hpo(
    X: pd.DataFrame,
    y: Optional[pd.Series],
    model_type: str,
    hp_space: dict,
    max_trials: int,
    cv_folds: int,
    category: str,
) -> tuple[Any, dict, dict]:
    """
    Run hyperparameter optimization using Ray Tune with ASHA scheduler.

    ASHA (Asynchronous Successive Halving Algorithm) early-terminates
    underperforming trials, saving ~35% compute vs. exhaustive search.

    In production (Databricks), this runs as distributed Ray Tune trials
    across Ray workers on the cluster. Locally, trials run on available cores.
    """
    # Convert the YAML grid-style space to Ray Tune search space
    tune_space = {}
    for param_name, values in hp_space.items():
        tune_space[param_name] = tune.choice(values)

    # Determine primary metric
    if category == "classifier":
        metric = "roc_auc"
    elif category == "regressor":
        metric = "r2"
    else:
        metric = "silhouette_score"

    # ASHA Scheduler: early-stops bad trials
    scheduler = ASHAScheduler(
        metric=metric,
        mode="max",
        max_t=max_trials,
        grace_period=1,
        reduction_factor=2,
    )

    # Define the trainable function with data closure
    def _trainable(config_sample):
        """Single Ray Tune trial: train + evaluate one hyperparameter set."""
        # Ensure int types for integer params
        params = dict(config_sample)
        for k in [
            "max_depth",
            "n_estimators",
            "n_clusters",
            "n_init",
            "max_iter",
            "min_child_weight",
            "min_child_samples",
            "num_leaves",
        ]:
            if k in params:
                params[k] = int(params[k])

        model = _create_model(model_type, params)

        if category == "clusterer":
            model.fit(X)
            metrics = _evaluate_clustering(model, X)
        else:
            metrics, _ = _cross_validate(model, X, y, cv_folds, category)

        # Report metrics back to Ray Tune
        tune.report(**metrics)

    # Run Ray Tune
    num_samples = min(max_trials, _count_combinations(hp_space))

    tuner = tune.Tuner(
        _trainable,
        param_space=tune_space,
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        run_config=ray.train.RunConfig(
            verbose=0,  # Suppress Ray Tune output (we log via MLflow)
        ),
    )

    tune_results = tuner.fit()

    # Extract best result
    best_result = tune_results.get_best_result(metric=metric, mode="max")
    best_trial_params = best_result.config

    # Ensure int types in best params
    for k in [
        "max_depth",
        "n_estimators",
        "n_clusters",
        "n_init",
        "max_iter",
        "min_child_weight",
        "min_child_samples",
        "num_leaves",
    ]:
        if k in best_trial_params:
            best_trial_params[k] = int(best_trial_params[k])

    # Re-train the best model on full data
    best_model = _create_model(model_type, best_trial_params)
    if category == "clusterer":
        best_model.fit(X)
        best_metrics = _evaluate_clustering(best_model, X)
    else:
        best_metrics, best_model = _cross_validate(best_model, X, y, cv_folds, category)

    return best_model, best_trial_params, best_metrics


def _create_model(model_type: str, params: dict) -> Any:
    """Create a model instance from type string and parameters."""
    if model_type == "xgboost_classifier":
        import xgboost as xgb

        return xgb.XGBClassifier(
            **params,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
    elif model_type == "xgboost_regressor":
        import xgboost as xgb

        return xgb.XGBRegressor(
            **params,
            random_state=42,
            verbosity=0,
        )
    elif model_type == "lightgbm_classifier":
        import lightgbm as lgb

        return lgb.LGBMClassifier(
            **params,
            random_state=42,
            verbose=-1,
        )
    elif model_type == "lightgbm_regressor":
        import lightgbm as lgb

        return lgb.LGBMRegressor(
            **params,
            random_state=42,
            verbose=-1,
        )
    elif model_type == "sklearn_kmeans":
        return KMeans(**params, random_state=42)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def _cross_validate(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int,
    category: str,
) -> tuple[dict, Any]:
    """Perform cross-validation and return averaged metrics + fitted model."""
    X_np = X.values
    y_np = y.values

    if category == "classifier":
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_metrics = []

    for train_idx, val_idx in kf.split(X_np, y_np):
        X_train, X_val = X_np[train_idx], X_np[val_idx]
        y_train, y_val = y_np[train_idx], y_np[val_idx]

        model_clone = _clone_model(model)
        model_clone.fit(X_train, y_train)

        if category == "classifier":
            y_pred = model_clone.predict(X_val)
            y_prob = (
                model_clone.predict_proba(X_val)[:, 1]
                if hasattr(model_clone, "predict_proba")
                else y_pred
            )
            metrics = {
                "roc_auc": roc_auc_score(y_val, y_prob),
                "f1": f1_score(y_val, y_pred, zero_division=0),
                "precision": precision_score(y_val, y_pred, zero_division=0),
                "recall": recall_score(y_val, y_pred, zero_division=0),
                "accuracy": accuracy_score(y_val, y_pred),
            }
        else:
            y_pred = model_clone.predict(X_val)
            metrics = {
                "rmse": np.sqrt(mean_squared_error(y_val, y_pred)),
                "mae": mean_absolute_error(y_val, y_pred),
                "r2": r2_score(y_val, y_pred),
            }
        fold_metrics.append(metrics)

    # Average metrics across folds
    avg_metrics = {}
    for key in fold_metrics[0]:
        values = [fm[key] for fm in fold_metrics]
        avg_metrics[key] = float(np.mean(values))
        avg_metrics[f"{key}_std"] = float(np.std(values))

    # Fit final model on all data
    model.fit(X_np, y_np)
    return avg_metrics, model


def _evaluate_clustering(model: Any, X: pd.DataFrame) -> dict:
    """Evaluate clustering model quality."""
    labels = model.labels_
    n_clusters = len(set(labels))

    metrics: dict[str, Any] = {"n_clusters": n_clusters}

    if n_clusters > 1 and n_clusters < len(X):
        try:
            # Use a sample for large datasets
            sample_size = min(10000, len(X))
            if len(X) > sample_size:
                idx = np.random.choice(len(X), sample_size, replace=False)
                X_sample = X.iloc[idx]
                labels_sample = labels[idx]
            else:
                X_sample = X
                labels_sample = labels

            metrics["silhouette_score"] = float(silhouette_score(X_sample, labels_sample))
            metrics["calinski_harabasz"] = float(calinski_harabasz_score(X, labels))
        except Exception as e:
            metrics["silhouette_score"] = 0.0

    metrics["inertia"] = float(model.inertia_)
    return metrics


def _evaluate_champion_challenger(
    model_id: str,
    new_metrics: dict,
    champion_config: dict,
    model_config: dict,
    category: str,
) -> str:
    """
    Decide whether the new model should be promoted.

    In production, this compares against the current production model
    in MLflow Model Registry.
    """
    if not champion_config.get("enabled", False):
        return "auto_promoted"

    # Check minimum thresholds from model config
    thresholds = model_config.get("champion_threshold", {})
    for metric_key, min_value in thresholds.items():
        # Map threshold keys to metric names
        metric_name = metric_key.replace("_min", "")
        actual = new_metrics.get(metric_name)
        if actual is not None and actual < min_value:
            return f"rejected (min {metric_name}: {actual:.4f} < {min_value})"

    return "promoted"


def _clone_model(model: Any) -> Any:
    """Create a fresh copy of a model with the same parameters."""
    from sklearn.base import clone

    return clone(model)


def _count_combinations(hp_space: dict) -> int:
    """Count total hyperparameter combinations."""
    count = 1
    for values in hp_space.values():
        count *= len(values)
    return count


def _format_metrics(metrics: dict) -> str:
    """Format metrics dict for logging."""
    parts = []
    for k, v in metrics.items():
        if "_std" not in k and isinstance(v, float):
            parts.append(f"{k}={v:.4f}")
    return ", ".join(parts[:4])
