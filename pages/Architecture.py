"""
ModelFactory — System Architecture Page
───────────────────────────────────────────────────────────────────────────────
Renders a Graphviz diagram of the pipeline architecture with component
descriptions, data flow, and tech stack highlights.
"""

from __future__ import annotations

import streamlit as st

# Graphviz for the architecture diagram
try:
    import graphviz
except ImportError:
    st.error(
        "The `graphviz` Python package is required. "
        "Install it with: `pip install graphviz`"
    )
    st.stop()


st.set_page_config(
    page_title="Architecture — ModelFactory",
    page_icon="🏗️",
    layout="wide",
)

st.title("🏗️ System Architecture")
st.markdown(
    "End-to-end MLOps pipeline with **Ray** distributed computing, "
    "**MLflow** experiment tracking, and **Streamlit** frontend."
)

# ── Architecture Diagram ─────────────────────────────────────────────────────

st.subheader("Pipeline Overview")

diagram = graphviz.Digraph(
    name="mlops_architecture",
    comment="ModelFactory MLOps Pipeline Architecture",
    format="svg",
)
diagram.attr(
    rankdir="LR",
    splines="ortho",
    fontname="Helvetica Neue",
    fontsize="12",
    bgcolor="transparent",
    dpi="150",
)
diagram.attr("node", shape="box", style="filled,rounded", fontname="Helvetica Neue", fontsize="10")
diagram.attr("edge", fontname="Helvetica Neue", fontsize="9", penwidth="1.2")

# Define colours
COLOUR_DATA = "#A8D8EA"       # light blue
COLOUR_FEATURE = "#F3B0C3"    # pink
COLOUR_TRAIN = "#C3AED6"      # lavender
COLOUR_INFERENCE = "#B5EAD7"  # mint
COLOUR_MONITOR = "#FFDAC1"    # peach
COLOUR_TRACKING = "#FFE156"   # yellow
COLOUR_STORAGE = "#D4A5A5"    # rose
COLOUR_INFRA = "#B0BEC5"      # blue-grey

# ── Data Sources ─────────────────────────────────────────────────────────────
with diagram.subgraph(name="cluster_data") as sg:
    sg.attr(
        label="Data Layer",
        style="filled",
        fillcolor="#F0F4F8",
        fontsize="11",
        fontname="Helvetica Neue",
    )
    sg.node("synthetic", "Synthetic Data\nGenerator", fillcolor=COLOUR_DATA)
    sg.node("feature_store", "Feature Store\n(Parquet / Databricks)", fillcolor=COLOUR_DATA)

# ── Feature Engineering ──────────────────────────────────────────────────────
with diagram.subgraph(name="cluster_feature") as sg:
    sg.attr(label="Feature Engineering", style="filled", fillcolor="#F0F4F8", fontsize="11")
    sg.node("feature_pipeline", "Spark / Pandas\nTransform Pipeline", fillcolor=COLOUR_FEATURE)
    sg.node("feature_selection", "Feature Selection\n(Variance / Correlation)", fillcolor=COLOUR_FEATURE)

# ── Training ─────────────────────────────────────────────────────────────────
with diagram.subgraph(name="cluster_train") as sg:
    sg.attr(label="Model Training", style="filled", fillcolor="#F0F4F8", fontsize="11")
    sg.node("ray_tune", "Ray Tune\nHyperparameter Opt.", fillcolor=COLOUR_TRAIN)
    sg.node("models", "Model Registry\n(12 Model Variants)", fillcolor=COLOUR_TRAIN)
    sg.node("mlflow_train", "MLflow Tracking\n(Experiments + Runs)", fillcolor=COLOUR_TRACKING)

# ── Inference ────────────────────────────────────────────────────────────────
with diagram.subgraph(name="cluster_infer") as sg:
    sg.attr(label="Inference", style="filled", fillcolor="#F0F4F8", fontsize="11")
    sg.node("ray_actors", "Ray Actors\nParallel Inference", fillcolor=COLOUR_INFERENCE)
    sg.node("predictions", "Prediction Store\n(Parquet)", fillcolor=COLOUR_INFERENCE)

# ── Monitoring ───────────────────────────────────────────────────────────────
with diagram.subgraph(name="cluster_monitor") as sg:
    sg.attr(label="Monitoring", style="filled", fillcolor="#F0F4F8", fontsize="11")
    sg.node("drift", "Drift Detection\n(PSI / KS Test)", fillcolor=COLOUR_MONITOR)
    sg.node("alerts", "Alert Engine\n(Threshold-based)", fillcolor=COLOUR_MONITOR)

# ── Governance ───────────────────────────────────────────────────────────────
diagram.node("compliance", "Governance\n(PII / Audit / GDPR)", fillcolor=COLOUR_STORAGE)

# ── Infrastructure ───────────────────────────────────────────────────────────
with diagram.subgraph(name="cluster_infra") as sg:
    sg.attr(label="Infrastructure", style="filled", fillcolor="#F0F4F8", fontsize="11")
    sg.node("tf", "Terraform\n(Databricks)", fillcolor=COLOUR_INFRA)
    sg.node("docker", "Docker\n(MLflow + Pipeline)", fillcolor=COLOUR_INFRA)
    sg.node("ci", "GitHub Actions\n(CI/CD)", fillcolor=COLOUR_INFRA)

# ── Frontend ─────────────────────────────────────────────────────────────────
diagram.node("streamlit", "Streamlit UI\n(Dashboard + Viz)", fillcolor="#AED6F1", shape="box3d")

# ── Edges — Data Flow ────────────────────────────────────────────────────────
diagram.edge("synthetic", "feature_pipeline", label="raw data")
diagram.edge("feature_store", "feature_pipeline", label="stored features")
diagram.edge("feature_pipeline", "feature_selection", label="transformed")
diagram.edge("feature_selection", "ray_tune", label="features")
diagram.edge("feature_selection", "ray_actors", label="features")
diagram.edge("ray_tune", "models", label="best params")
diagram.edge("ray_tune", "mlflow_train", label="log metrics")
diagram.edge("models", "ray_actors", label="load artifacts")
diagram.edge("ray_actors", "predictions", label="predictions")
diagram.edge("predictions", "drift", label="reference")
diagram.edge("feature_selection", "drift", label="baseline")
diagram.edge("drift", "alerts", label="drift scores")
diagram.edge("alerts", "streamlit", label="notifications")
diagram.edge("mlflow_train", "streamlit", label="experiments")
diagram.edge("predictions", "streamlit", label="results")

# ── Edges — Governance ───────────────────────────────────────────────────────
diagram.edge("compliance", "synthetic", label="audit trail")
diagram.edge("compliance", "models", label="model lineage")

# ── Edges — Infra overlays ───────────────────────────────────────────────────
diagram.edge("docker", "mlflow_train", style="dashed", arrowhead="none", label="containerises")
diagram.edge("tf", "feature_store", style="dashed", arrowhead="none", label="provisioned by")
diagram.edge("ci", "docker", style="dashed", arrowhead="none", label="triggers")

st.graphviz_chart(diagram, use_container_width=True)

st.caption(
    "**Data flow** (solid arrows) moves left-to-right from ingestion through "
    "monitoring. **Infrastructure dependencies** (dashed arrows) show deployment "
    "relationships."
)

# ── Component Breakdown ──────────────────────────────────────────────────────

st.divider()
st.subheader("🔧 Component Breakdown")

cols = st.columns(3)

with cols[0]:
    with st.container(border=True):
        st.markdown("#### 📥 Data Ingestion")
        st.markdown(
            "- Generates **synthetic** tabular data (regression, classification, "
            "clustering targets)\n"
            "- Configurable row count via `pipeline_config.yaml`\n"
            "- Output: Pandas DataFrames cached in-memory"
        )
    with st.container(border=True):
        st.markdown("#### ⚙️ Feature Engineering")
        st.markdown(
            "- **Pipeline**: StandardScaler → PolynomialFeatures → SelectKBest\n"
            "- Low-variance & high-correlation filters\n"
            "- Returns engineered DataFrame + fitted pipeline"
        )

with cols[1]:
    with st.container(border=True):
        st.markdown("#### 🧠 Model Training (Ray Tune)")
        st.markdown(
            "- **Distributed HPO** via Ray Tune (ASHA scheduler)\n"
            "- 12 model variants: Logistic Regression, Random Forest, "
            "Gradient Boosting, SVC, KNN, MLP, XGBoost, LightGBM, "
            "Extra Trees, Ridge, Lasso, K-Means\n"
            "- MLflow auto-logging per trial\n"
            "- Winner saved as MLflow artifact"
        )
    with st.container(border=True):
        st.markdown("#### ⚡ Batch Inference")
        st.markdown(
            "- Ray **Actor**-based parallel inference\n"
            "- Loads best artifact from MLflow\n"
            "- Predictions stored as Parquet\n"
            "- Configurable batch size"
        )

with cols[2]:
    with st.container(border=True):
        st.markdown("#### 📊 Model Monitoring")
        st.markdown(
            "- **Drift detection**: Population Stability Index (PSI) + "
            "Kolmogorov–Smirnov (KS)\n"
            "- Threshold-based alerting\n"
            "- Works with both features + predictions"
        )
    with st.container(border=True):
        st.markdown("#### 📜 Governance")
        st.markdown(
            "- **PII scanning** on training data\n"
            "- Audit trail logging\n"
            "- Model snapshot & lineage DAG\n"
            "- GDPR erasure simulation"
        )

# ── Tech Stack Table ─────────────────────────────────────────────────────────

st.divider()
st.subheader("🛠️ Tech Stack")

tech = {
    "Category": [
        "Orchestration",
        "Distributed Compute",
        "Experiment Tracking",
        "Feature Store",
        "Infrastructure",
        "CI/CD",
        "Frontend",
    ],
    "Technology": [
        "Python 3.12 + Custom DAG",
        "Ray Core / Ray Tune / Ray Actors",
        "MLflow Tracking Server",
        "Local Parquet + Databricks Feature Store",
        "Terraform + Docker Compose",
        "GitHub Actions",
        "Streamlit",
    ],
    "Status": [
        "✅ Implemented",
        "✅ Implemented",
        "✅ Self-hosted MLflow",
        "✅ Both backends",
        "✅ Terraform for Databricks",
        "✅ Lint → Test → Deploy",
        "✅ Interactive Dashboard",
    ],
}
import pandas as pd
st.dataframe(
    pd.DataFrame(tech),
    use_container_width=True,
    hide_index=True,
)

# ── Pipeline Configuration Summary ───────────────────────────────────────────

st.divider()
st.subheader("📄 Pipeline Configuration")

from src.utils import load_config, load_model_registry

try:
    config = load_config()
    reg = load_model_registry()

    conf_cols = st.columns(2)
    with conf_cols[0]:
        st.markdown("**Pipeline Config**")
        st.json(
            {
                "data": config.get("data", {}),
                "training": config.get("training", {}),
                "monitoring": config.get("monitoring", {}),
            }
        )
    with conf_cols[1]:
        st.markdown("**Model Registry**")
        models = reg.get("models", [])
        st.json(
            {
                "models": [
                    {"name": m["name"], "type": m.get("type"),
                     "task": m.get("task")}
                    for m in models
                ]
            }
        )
except Exception:
    st.info("Load configs from project root to see live configuration.")
