# ExperimentOS

A single-user, local Streamlit MVP that turns a CSV into trained, ranked, and
logged ML baselines — then runs a small **deterministic** improvement loop. No
LLM or external API calls anywhere.

LangGraph is used purely as a deterministic DAG / state-machine orchestrator.
Every node is plain Python; the one conditional edge routes on flags we compute
ourselves (imbalance? skew? round < 2?). This gives us resumable, inspectable,
cyclic control flow without hand-rolling a `while` loop.

## What it does

1. Upload a CSV and pick a target column.
2. Auto-detect the task type (classification vs regression).
3. Profile the data for **class imbalance** and **numeric skew**.
4. Train candidate models (Logistic/Linear, RandomForest, XGBoost or
   GradientBoosting fallback), logging every run to local MLflow.
5. Rank them on a leaderboard (`f1_weighted` for classification, `rmse` for
   regression).
6. Run up to **2** deterministic improvement rounds:
   - imbalance → `class_weight="balanced"` (sample weights for boosters)
   - skew → `RobustScaler` instead of `StandardScaler`
7. Show the leaderboard, a comparison chart, the improvement log, and a
   downloadable self-contained markdown report.

## Setup

```bash
cd experimentos
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

`xgboost` is optional — if it fails to install/import, the app automatically
falls back to scikit-learn's `GradientBoosting` estimators.

## Run

```bash
streamlit run app.py
```

## Inspect MLflow runs

Every model/round is logged to a local file store at `./mlruns`. From the
project root:

```bash
mlflow ui
```

then open the printed URL (default http://127.0.0.1:5000).

## Reproducibility

`random_state=42` is fixed everywhere (splits and models), so runs are
reproducible. The improvement loop is deterministic: it tries each applicable
fix once and reports what happened — it does **not** chase metric improvement.

## Project layout

```
experimentos/
  app.py                 # Streamlit UI
  graph/
    state.py             # ExperimentState TypedDict
    nodes.py             # node functions + router
    workflow.py          # builds + compiles the StateGraph
  ml/
    preprocessing.py     # ColumnTransformer builder
    models.py            # candidate model registry
    metrics.py           # metric computation
    diagnostics.py       # task type / imbalance / skew detection
    report.py            # markdown report with embedded chart
  mlflow_utils.py        # init_experiment / log_run
```

## Out of scope

Hyperparameter tuning/AutoML, any LLM or external API, multi-user auth, cloud
deployment, remote MLflow, and more than 2 improvement rounds.
