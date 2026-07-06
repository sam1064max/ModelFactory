"""
ModelFactory — Streamlit Frontend
───────────────────────────────────────────────────────────────────────────────
Interactive dashboard for the MLOps pipeline: monitor stage execution, inspect
metrics, and explore the system architecture.

Usage:
    streamlit run app.py
"""

from __future__ import annotations

import threading
import time
from queue import Queue
from typing import Any

import streamlit as st

from frontend.pipeline_runner import (
    STAGES,
    STAGE_LABELS,
    STAGE_RUNNING,
    STAGE_DONE,
    STAGE_FAILED,
    StreamlitPipelineRunner,
)

# ── Page Configuration ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="ModelFactory",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp header {display: none;}
    .block-container {padding-top: 1.5rem;}
    .stStatusWidget {border-radius: 8px;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Session State ────────────────────────────────────────────────────────────

def init_state() -> None:
    if "runner" not in st.session_state:
        st.session_state.runner = None
    if "pipeline_running" not in st.session_state:
        st.session_state.pipeline_running = False
    if "pipeline_done" not in st.session_state:
        st.session_state.pipeline_done = False
    if "results" not in st.session_state:
        st.session_state.results = None
    if "logs" not in st.session_state:
        st.session_state.logs = []
    if "stage_status" not in st.session_state:
        st.session_state.stage_status = {s: "idle" for s in STAGES}
    if "stage_data" not in st.session_state:
        st.session_state.stage_data = {s: {} for s in STAGES}
    if "stage_times" not in st.session_state:
        st.session_state.stage_times = {}
    if "progress" not in st.session_state:
        st.session_state.progress = 0.0
    if "error" not in st.session_state:
        st.session_state.error = None


init_state()


# ── Pipeline Callbacks ───────────────────────────────────────────────────────

def _on_log(msg: str) -> None:
    st.session_state.logs.append(msg)


def _on_stage(name: str, status: str, data: dict) -> None:
    st.session_state.stage_status[name] = status
    if data:
        st.session_state.stage_data[name] = data


def _on_progress(value: float) -> None:
    st.session_state.progress = value


def _run_pipeline() -> None:
    """Background thread target — runs the full pipeline."""
    try:
        runner = StreamlitPipelineRunner()
        # Apply sidebar overrides
        if st.session_state.get("override_rows"):
            runner.set_training_rows(st.session_state.override_rows)
        if st.session_state.get("override_infer"):
            runner.set_inference_rows(st.session_state.override_infer)
        if st.session_state.get("override_models"):
            runner.set_num_models(st.session_state.override_models)

        st.session_state.runner = runner
        results = runner.run(
            on_log=_on_log,
            on_stage=_on_stage,
            on_progress=_on_progress,
        )
        st.session_state.results = results
        if results:
            st.session_state.stage_times = results.get("stage_times", {})
        st.session_state.pipeline_done = True
    except Exception as exc:
        st.session_state.error = str(exc)
    finally:
        st.session_state.pipeline_running = False


# ── Sidebar — Configuration ──────────────────────────────────────────────────

with st.sidebar:
    st.title("🏭 ModelFactory")
    st.caption("MLOps Pipeline Demo — Ray + MLflow")

    st.divider()
    st.subheader("⚙️ Configuration Overrides")

    with st.container(border=True):
        st.session_state.override_rows = st.number_input(
            "Training rows",
            min_value=100, max_value=50000, value=5000, step=500,
            disabled=st.session_state.pipeline_running,
            help="Rows of synthetic training data to generate.",
        )
        st.session_state.override_infer = st.number_input(
            "Inference rows",
            min_value=50, max_value=10000, value=1000, step=100,
            disabled=st.session_state.pipeline_running,
            help="Rows of synthetic inference data.",
        )
        st.session_state.override_models = st.slider(
            "Number of models",
            min_value=1, max_value=12, value=6,
            disabled=st.session_state.pipeline_running,
            help="Models to train (limited subset for faster demo).",
        )

    st.divider()

    st.subheader("▶️ Control")
    run_disabled = st.session_state.pipeline_running or st.session_state.pipeline_done
    if st.button(
        "🚀 Run Full Pipeline",
        type="primary",
        use_container_width=True,
        disabled=run_disabled,
    ):
        # Reset state
        st.session_state.pipeline_running = True
        st.session_state.pipeline_done = False
        st.session_state.logs = []
        st.session_state.stage_status = {s: "idle" for s in STAGES}
        st.session_state.stage_data = {s: {} for s in STAGES}
        st.session_state.stage_times = {}
        st.session_state.progress = 0.0
        st.session_state.error = None
        st.session_state.results = None

        thread = threading.Thread(target=_run_pipeline, daemon=True)
        thread.start()
        st.rerun()

    if st.session_state.pipeline_done:
        if st.button("🔄 Reset", use_container_width=True):
            for key in ["pipeline_running", "pipeline_done", "results",
                        "logs", "stage_status", "stage_data", "stage_times",
                        "error", "runner"]:
                if key in st.session_state:
                    if key in ("stage_status", "stage_data"):
                        st.session_state[key] = {
                            s: {} if key == "stage_data" else "idle"
                            for s in STAGES
                        }
                    elif key == "logs":
                        st.session_state[key] = []
                    elif key == "stage_times":
                        st.session_state[key] = {}
                    elif key == "progress":
                        st.session_state[key] = 0.0
                    else:
                        st.session_state[key] = None
            st.rerun()

    st.divider()
    st.caption("v1.2.0 • [GitHub](https://github.com/sam1064max/ModelFactory)")


# ── Main Area — Pipeline Visualization ───────────────────────────────────────

col1, col2 = st.columns([7, 3])

with col1:
    st.subheader("📊 Pipeline Execution")
with col2:
    if st.session_state.pipeline_running:
        st.progress(st.session_state.progress, text="Overall progress")

# ── Stage Cards ──────────────────────────────────────────────────────────────

stage_cols = st.columns(len(STAGES))

for idx, (stage_col, stage_name) in enumerate(zip(stage_cols, STAGES)):
    label = STAGE_LABELS[stage_name]
    status = st.session_state.stage_status.get(stage_name, "idle")
    data = st.session_state.stage_data.get(stage_name, {})
    elapsed = st.session_state.stage_times.get(stage_name)

    with stage_col:
        # Status icon
        if status == STAGE_RUNNING:
            icon, bg = "🔄", "#FFF3CD"
        elif status == STAGE_DONE:
            icon, bg = "✅", "#D4EDDA"
        elif status == STAGE_FAILED:
            icon, bg = "❌", "#F8D7DA"
        else:
            icon, bg = "⏸️", "#E2E3E5"

        st.markdown(
            f"<div style='background:{bg};padding:0.75rem;"
            f"border-radius:8px;text-align:center;min-height:120px'>"
            f"<div style='font-size:1.5rem'>{icon}</div>"
            f"<div style='font-weight:600;font-size:0.85rem;"
            f"margin-top:0.3rem'>{label}</div>",
            unsafe_allow_html=True,
        )

        if data:
            for k, v in data.items():
                label_k = k.replace("_", " ").title()
                if isinstance(v, float):
                    st.metric(label_k, f"{v:.2f}")
                else:
                    st.metric(label_k, f"{v:,}" if isinstance(v, int) else v)

        if elapsed is not None:
            st.caption(f"⏱ {elapsed:.1f}s")

        st.markdown("</div>", unsafe_allow_html=True)


# ── Results Dashboard ────────────────────────────────────────────────────────

if st.session_state.results:
    st.divider()
    st.subheader("📈 Pipeline Results")

    m = st.session_state.results.get("summary_metrics", {})
    if m:
        kpi_row = st.columns(6)
        with kpi_row[0]:
            st.metric("Models Trained", m.get("models_trained", 0))
        with kpi_row[1]:
            st.metric("Features Engineered", m.get("num_features", 0))
        with kpi_row[2]:
            st.metric("Inference Records", f"{m.get('inference_records', 0):,}")
        with kpi_row[3]:
            avg_acc = m.get("avg_accuracy", 0)
            st.metric("Avg ROC-AUC / R²", f"{avg_acc:.3f}")
        with kpi_row[4]:
            st.metric("Drifted Features", m.get("features_drifted", 0))
        with kpi_row[5]:
            total = m.get("total_time", 0)
            st.metric("Total Time", f"{total:.1f}s")

    # ── Model Training Details ──────────────────────────────────────────
    training_results = st.session_state.results.get("training_results", {})
    if training_results:
        st.divider()
        st.subheader("🤖 Model Training Results")
        model_rows = []
        for model_name, result in training_results.items():
            status = result.get("status", "unknown")
            best_metrics = result.get("best_metrics", {})
            metric_str = "; ".join(
                f"{k}={v:.4f}" for k, v in best_metrics.items()
            )
            model_rows.append({
                "Model": model_name,
                "Status": "✅" if status == "success" else "❌",
                "Best Params": str(result.get("best_params", {})),
                "Metrics": metric_str or "-",
            })
        st.dataframe(model_rows, use_container_width=True, hide_index=True)

    # ── Monitoring Details ──────────────────────────────────────────────
    monitoring_results = st.session_state.results.get("monitoring_results", {})
    drift = monitoring_results.get("feature_drift", {})
    alerts = monitoring_results.get("alerts", [])
    if drift:
        st.divider()
        tab1, tab2 = st.tabs(["🔍 Feature Drift", "🔔 Alerts"])
        with tab1:
            drift_rows = []
            for feat, info in drift.items():
                drift_rows.append({
                    "Feature": feat,
                    "Drifted": "⚠️" if info.get("is_drifted", False) else "✅",
                    "PSI": f"{info.get('psi', 0):.4f}",
                    "P-Value": f"{info.get('ks_pvalue', 0):.4f}",
                })
            if drift_rows:
                st.dataframe(drift_rows, use_container_width=True, hide_index=True)
        with tab2:
            if alerts:
                st.warning(f"{len(alerts)} alert(s) generated")
                for alert in alerts[:10]:
                    st.markdown(f"- {alert}")
            else:
                st.success("No alerts — pipeline healthy")


# ── Live Logs ────────────────────────────────────────────────────────────────

st.divider()
with st.expander("📋 Pipeline Logs", expanded=bool(st.session_state.pipeline_running)):
    log_container = st.empty()
    logs_text = "\n".join(st.session_state.logs[-100:])
    log_container.code(logs_text, language="log")

# ── Error Display ────────────────────────────────────────────────────────────

if st.session_state.error:
    st.error(f"Pipeline failed: {st.session_state.error}")


# ── Auto-refresh while running ───────────────────────────────────────────────

if st.session_state.pipeline_running:
    time.sleep(0.3)
    st.rerun()
