"""ExperimentOS — Streamlit entrypoint.

UI only: upload a CSV, pick a target, run the deterministic LangGraph workflow,
and view the leaderboard, comparison chart, improvement log, and report.
"""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from graph.workflow import build_workflow, initial_state
from ml import metrics as metrics_mod
from mlflow_utils import init_experiment

st.set_page_config(page_title="ExperimentOS", layout="wide")
st.title("🧪 ExperimentOS")
st.caption(
    "Upload a CSV, pick a target, and get baseline models trained, logged to "
    "MLflow, ranked, and run through a deterministic improvement loop — no LLMs."
)

uploaded = st.file_uploader("Upload a CSV", type=["csv"])

if uploaded is not None:
    import pandas as pd

    df = pd.read_csv(uploaded)
    st.subheader("Preview")
    st.dataframe(df.head())

    target_column = st.selectbox("Target column", options=list(df.columns))

    if st.button("Run Experiment", type="primary"):
        with st.spinner("Running experiment (2 rounds × candidate models)…"):
            experiment_id = init_experiment(uploaded.name)
            graph = build_workflow()
            state = initial_state(df, target_column, experiment_id)
            final_state = graph.invoke(state)

        st.success("Experiment complete.")

        task_type = final_state["task_type"]
        primary = metrics_mod.PRIMARY_METRIC[task_type]
        st.info(f"Detected task type: **{task_type}** · primary metric: `{primary}`")

        if final_state.get("split_warning"):
            st.warning(final_state["split_warning"])

        leaderboard = final_state["leaderboard"]

        # Leaderboard
        st.subheader("🏆 Leaderboard")
        st.dataframe(leaderboard, use_container_width=True)

        # Comparison chart
        st.subheader(f"📊 {primary} comparison")
        chart_df = leaderboard.copy()
        chart_df["label"] = (
            chart_df["model"] + " (r" + chart_df["round"].astype(str) + ")"
        )
        fig = px.bar(
            chart_df,
            x="label",
            y=primary,
            color="stage",
            title=f"{primary} by model / round",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Improvement log
        with st.expander("🔧 Improvement log", expanded=True):
            if final_state["improvement_log"]:
                for note in final_state["improvement_log"]:
                    st.write(f"- {note}")
            else:
                st.write("No improvement rounds were applied.")

        # Report download
        st.download_button(
            "⬇️ Download markdown report",
            data=final_state["report_markdown"],
            file_name="experimentos_report.md",
            mime="text/markdown",
        )
