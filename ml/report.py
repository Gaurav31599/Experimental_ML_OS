"""Self-contained markdown report builder.

The comparison chart is rendered with matplotlib and embedded as a base64 PNG so
the resulting ``.md`` file is a single portable download with no side-car images.
"""

from __future__ import annotations

import base64
import io

import matplotlib

matplotlib.use("Agg")  # headless backend — no display required
import matplotlib.pyplot as plt
import pandas as pd

from ml import metrics as metrics_mod


def _chart_base64(leaderboard: pd.DataFrame, task_type: str) -> str:
    """Render the primary-metric comparison as a base64-encoded PNG data URI."""
    primary = metrics_mod.PRIMARY_METRIC[task_type]
    labels = [
        f"{row.model} (r{row.round})" for row in leaderboard.itertuples()
    ]
    values = leaderboard[primary].tolist()

    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.5 * len(labels))))
    ax.barh(labels, values, color="#4C78A8")
    ax.set_xlabel(primary)
    ax.set_title(f"{primary} by model / round")
    ax.invert_yaxis()  # best (first row) on top
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def build_markdown_report(state) -> str:
    """Build the full markdown report string from the final state."""
    df = state["raw_df"]
    leaderboard = state["leaderboard"]
    task_type = state["task_type"]
    primary = metrics_mod.PRIMARY_METRIC[task_type]

    lines: list[str] = []
    lines.append("# ExperimentOS Report\n")

    # Dataset summary
    lines.append("## Dataset Summary\n")
    lines.append(f"- **Rows:** {len(df)}")
    lines.append(f"- **Columns:** {df.shape[1]}")
    lines.append(f"- **Target column:** `{state['target_column']}`")
    lines.append(f"- **Task type:** {task_type}\n")

    # Detected issues
    lines.append("## Detected Issues\n")
    if state["imbalance_detected"]:
        ratio = state["imbalance_ratio"] or 0.0
        lines.append(f"- **Class imbalance:** ratio {ratio:.1f}:1")
    else:
        lines.append("- **Class imbalance:** none detected")
    if state["skew_detected"]:
        lines.append(f"- **Skewed numeric columns:** {', '.join(state['skewed_cols'])}")
    else:
        lines.append("- **Numeric skew:** none detected")
    lines.append("")

    # Leaderboard table
    lines.append("## Leaderboard\n")
    if leaderboard is not None and not leaderboard.empty:
        table_df = leaderboard.drop(columns=["mlflow_run_id"], errors="ignore")
        lines.append(table_df.to_markdown(index=False))
    else:
        lines.append("_No results._")
    lines.append("")

    # Comparison chart
    lines.append(f"## {primary} Comparison\n")
    if leaderboard is not None and not leaderboard.empty:
        lines.append(f"![chart]({_chart_base64(leaderboard, task_type)})")
    lines.append("")

    # Improvement log
    lines.append("## Improvement Log\n")
    if state["improvement_log"]:
        for note in state["improvement_log"]:
            lines.append(f"- {note}")
    else:
        lines.append("- No improvement rounds were applied.")
    lines.append("")

    # Recommended model
    lines.append("## Recommended Model\n")
    if leaderboard is not None and not leaderboard.empty:
        best = leaderboard.iloc[0]
        lines.append(
            f"**{best['model']}** (round {best['round']}, stage `{best['stage']}`) "
            f"with {primary} = {best[primary]:.4f}."
        )
    else:
        lines.append("_No model to recommend._")
    lines.append("")

    return "\n".join(lines)
