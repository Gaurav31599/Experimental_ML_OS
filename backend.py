"""ExperimentOS FastAPI backend — SSE streaming wrapper for the LangGraph pipeline.

SETUP
-----
1. Copy this file into your `experimentos/` directory (alongside app.py).
2. Install the extra dependencies:

       pip install fastapi "uvicorn[standard]" python-multipart

3. Start the server from the experimentos/ directory:

       uvicorn backend:app --reload --port 8000

4. Open ExperimentOS.dc.html in the browser.
   It auto-detects the backend at http://localhost:8000.

ENDPOINTS
---------
  GET  /health              — liveness check
  POST /upload              — accept CSV → {run_id, columns, rows, filename}
  GET  /run/{run_id}        — SSE stream (query: ?target=column_name)
  GET  /report/{run_id}     — download the generated markdown report
"""

from __future__ import annotations

import asyncio
import io
import json
import math
import os
import sys
import time
import uuid
from typing import Any, AsyncGenerator

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from langgraph.graph import END, START, StateGraph

# ── Ensure graph modules are importable when run from the project root ───────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph import nodes as N
from graph.state import ExperimentState
from graph.workflow import initial_state
from mlflow_utils import init_experiment

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="ExperimentOS API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory stores (single-user local MVP) ─────────────────────────────────
_dfs:  dict[str, pd.DataFrame] = {}   # run_id → DataFrame
_meta: dict[str, dict]          = {}   # run_id → {filename, status, report}


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "ExperimentOS API",
        "status":  "ok",
        "docs":    "/docs",
        "health":  "/health",
    }


@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    """Accept a CSV and return run_id + column names for the frontend."""
    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(400, f"Cannot parse CSV: {exc}")

    run_id = uuid.uuid4().hex[:8]
    _dfs[run_id]  = df
    _meta[run_id] = {
        "filename": file.filename or "data.csv",
        "status":   "uploaded",
        "report":   None,
    }
    return {
        "run_id":   run_id,
        "columns":  list(df.columns),
        "rows":     len(df),
        "filename": file.filename,
    }


@app.get("/run/{run_id}")
async def run_pipeline(run_id: str, target: str):
    """Start the LangGraph pipeline and stream SSE events."""
    if run_id not in _dfs:
        raise HTTPException(404, "run_id not found — upload a CSV first")
    df = _dfs[run_id]
    if target not in df.columns:
        raise HTTPException(400, f"Column '{target}' is not in this dataset")

    _meta[run_id]["status"] = "running"
    loop  = asyncio.get_running_loop()
    queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

    def _thread():
        try:
            _pipeline_thread(run_id, df, target, _meta[run_id]["filename"], queue, loop)
        except Exception as exc:
            import traceback
            _put(loop, queue, "error", {
                "message":   str(exc),
                "traceback": traceback.format_exc(),
            })
            _put(loop, queue, "__close__", {})

    loop.run_in_executor(None, _thread)

    return StreamingResponse(
        _sse_gen(queue),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        },
    )


@app.get("/report/{run_id}")
def get_report(run_id: str):
    """Download the generated markdown report."""
    run = _meta.get(run_id)
    if not run or not run.get("report"):
        raise HTTPException(404, "Report not yet available")
    return Response(
        content=run["report"],
        media_type="text/markdown",
        headers={"Content-Disposition": f"attachment; filename=report_{run_id}.md"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# SSE helpers
# ─────────────────────────────────────────────────────────────────────────────

def _put(loop, queue: asyncio.Queue, event: str, data: Any) -> None:
    """Thread-safe push onto the asyncio event queue."""
    loop.call_soon_threadsafe(queue.put_nowait, (event, data))


async def _sse_gen(queue: asyncio.Queue) -> AsyncGenerator[str, None]:
    while True:
        event, data = await queue.get()
        if event == "__close__":
            break
        yield f"event: {event}\ndata: {json.dumps(data, default=_json_safe)}\n\n"


def _json_safe(obj: Any) -> Any:
    """Fallback JSON serializer for numpy / pandas types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON-serializable: {type(obj)}")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline thread (runs in a thread-pool worker)
# ─────────────────────────────────────────────────────────────────────────────

def _pipeline_thread(run_id, df, target, filename, queue, loop):
    emit = lambda ev, d:      _put(loop, queue, ev, d)
    log  = lambda text, k="info": emit("log", {"text": text, "type": k})

    # Mutable dict shared between wrapped node closures
    state_ref = {
        "prev_best":           None,   # best primary metric before current rank
        "pending_improvement": None,   # filled by apply_improvement, emitted after rank
    }

    t0 = time.time()
    try:
        exp_id = init_experiment(filename)
        graph  = _build_instrumented_graph(emit, log, filename, target, state_ref, run_id)
        init   = initial_state(df, target, exp_id)
        graph.invoke(init)

        elapsed = round(time.time() - t0, 1)
        _meta[run_id]["status"] = "complete"
        emit("done", {
            "elapsed":       elapsed,
            "experiment_id": exp_id,
            "run_id":        run_id,
        })

    except Exception as exc:
        import traceback
        _meta[run_id]["status"] = "error"
        emit("error", {
            "message":   str(exc),
            "traceback": traceback.format_exc(),
        })

    emit("__close__", {})


# ─────────────────────────────────────────────────────────────────────────────
# Instrumented graph builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_instrumented_graph(emit, log, filename, target, state_ref, run_id):
    """Wrap every node with SSE instrumentation, then compile a fresh graph."""

    def wrap(fn, name):
        def _w(state):
            emit("node_start", {"node": name})
            _pre_log(log, name, state, filename, target)
            updates = fn(state)
            # Local merge for inspection only — does not affect LangGraph reducers
            merged  = {**dict(state), **updates}
            _post_events(emit, log, name, merged, updates, state_ref, run_id)
            emit("node_complete", {"node": name})
            return updates
        return _w

    g = StateGraph(ExperimentState)
    g.add_node("ingest",             wrap(N.ingest_node,             "ingest"))
    g.add_node("detect_task_type",   wrap(N.detect_task_type_node,   "detect_task_type"))
    g.add_node("profile_data",       wrap(N.profile_data_node,       "profile_data"))
    g.add_node("build_preprocessor", wrap(N.build_preprocessor_node, "build_preprocessor"))
    g.add_node("train_baseline",     wrap(N.train_baseline_node,     "train_baseline"))
    g.add_node("rank",               wrap(N.rank_node,               "rank"))
    g.add_node("apply_improvement",  wrap(N.apply_improvement_node,  "apply_improvement"))
    g.add_node("generate_report",    wrap(N.generate_report_node,    "generate_report"))

    g.add_edge(START, "ingest")
    g.add_edge("ingest",             "detect_task_type")
    g.add_edge("detect_task_type",   "profile_data")
    g.add_edge("profile_data",       "build_preprocessor")
    g.add_edge("build_preprocessor", "train_baseline")
    g.add_edge("train_baseline",     "rank")
    g.add_conditional_edges(
        "rank", N.should_improve_router,
        {"improve": "apply_improvement", "report": "generate_report"},
    )
    g.add_edge("apply_improvement", "rank")
    g.add_edge("generate_report",   END)

    return g.compile()


# ─────────────────────────────────────────────────────────────────────────────
# Pre-node log messages
# ─────────────────────────────────────────────────────────────────────────────

def _pre_log(log, name, state, filename, target):
    rc = state.get("round_count", 0)
    msgs = {
        "ingest":              f"→ ingest: loading {filename}",
        "detect_task_type":    f"→ detect_task_type: scanning target \"{target}\"",
        "profile_data":         "→ profile_data: analyzing feature columns…",
        "build_preprocessor":   "→ build_preprocessor: assembling sklearn pipeline",
        "train_baseline":       "→ train_baseline: round 0 · 3 candidate models",
        "rank":                f"→ rank: rebuilding leaderboard (round {rc})",
        "apply_improvement":    "→ apply_improvement: applying fix…",
        "generate_report":      "→ generate_report: compiling markdown report…",
    }
    if name in msgs:
        log(msgs[name])


# ─────────────────────────────────────────────────────────────────────────────
# Post-node SSE events + logs
# ─────────────────────────────────────────────────────────────────────────────

def _post_events(emit, log, name, merged, updates, state_ref, run_id):

    if name == "ingest":
        df = merged.get("raw_df")
        if df is not None:
            log(f"  ↳ {len(df):,} rows × {len(df.columns)} cols", "dim")

    elif name == "detect_task_type":
        log(f"  ↳ task_type = {merged.get('task_type', '?')}", "dim")

    elif name == "profile_data":
        nc    = len(merged.get("numeric_cols",    []))
        cc    = len(merged.get("categorical_cols", []))
        imb   = merged.get("imbalance_detected", False)
        ratio = merged.get("imbalance_ratio")
        skew  = merged.get("skew_detected",  False)
        scols = merged.get("skewed_cols",    [])

        log(f"  ↳ {nc} numeric cols, {cc} categorical cols", "dim")
        if imb and ratio:
            log(f"  ↳ class imbalance {ratio:.1f}:1 detected", "warn")
        if skew and scols:
            log(f"  ↳ skew detected: {', '.join(str(c) for c in scols[:4])}", "warn")

        df      = merged.get("raw_df")
        maj_pct = round((ratio / (1.0 + ratio)) * 100) if ratio else 50

        emit("diagnostics", {
            "task_type":          merged.get("task_type"),
            "rows":               len(df) if df is not None else 0,
            "numeric_cols":       merged.get("numeric_cols",    []),
            "categorical_cols":   merged.get("categorical_cols", []),
            "imbalance_detected": bool(imb),
            "imbalance_ratio":    float(ratio) if ratio else None,
            "maj_pct":            maj_pct,
            "skew_detected":      bool(skew),
            "skewed_cols":        [str(c) for c in scols],
        })

    elif name == "build_preprocessor":
        log(f"  ↳ {merged.get('scaler_type','standard')}Scaler + OneHotEncoder", "dim")

    elif name in ("train_baseline", "apply_improvement"):
        results = updates.get("results", [])
        tt      = merged.get("task_type", "classification")
        primary = "f1_weighted" if tt == "classification" else "rmse"

        for r in results:
            m   = r.get("model_name", "?")
            val = r.get("metrics", {}).get(primary, 0.0)
            rid = (r.get("mlflow_run_id") or "")[:8]
            log(f"  ↳ {m:<22}  {primary}={val:.3f}  run={rid}", "dim")

        if name == "apply_improvement":
            imp_notes = updates.get("improvement_log", [])
            round_idx = updates.get("round_count", 1)
            fix_label = (
                "class_weight=balanced" if updates.get("imbalance_addressed") is True else
                "RobustScaler"          if updates.get("skew_addressed")       is True else
                "improvement"
            )
            state_ref["pending_improvement"] = {
                "round":  round_idx,
                "fix":    fix_label,
                "note":   imp_notes[-1] if imp_notes else "",
                "before": state_ref.get("prev_best"),
            }

        log(f"  ↳ {len(results)} runs logged to MLflow", "sys")

    elif name == "rank":
        lb = merged.get("leaderboard")
        if lb is None or lb.empty:
            return

        tt      = merged.get("task_type", "classification")
        primary = "f1_weighted" if tt == "classification" else "rmse"
        rows    = lb.to_dict(orient="records")

        for row in rows:
            for k, v in list(row.items()):
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    row[k] = None

        best   = rows[0].get(primary) if rows else None
        best_s = f"{best:.3f}" if best is not None else "?"
        model0 = rows[0].get("model", "?") if rows else "?"
        round0 = rows[0].get("round", 0)   if rows else 0
        log(f"  ↳ best {primary}={best_s} ({model0} r{round0})", "dim")

        emit("leaderboard", {
            "rows":           rows,
            "task_type":      tt,
            "primary_metric": primary,
            "prev_best":      state_ref.get("prev_best"),
            "current_best":   best,
        })

        # Emit improvement card now that we have the "after" value
        pending = state_ref.pop("pending_improvement", None)
        if pending:
            emit("improvement", {**pending, "after": best})

        state_ref["prev_best"] = best

    elif name == "generate_report":
        report = merged.get("report_markdown")
        if report:
            if run_id and run_id in _meta:
                _meta[run_id]["report"] = report
            emit("report", {"markdown": report})
            log("  ↳ report ready", "success")
