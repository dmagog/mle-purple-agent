"""
LLM loop: iteratively generates and executes Python code to solve an ML competition.
Uses OpenAI-compatible API (works with OpenRouter).
"""
import json
import os
import logging
import time

from openai import OpenAI, APIConnectionError, APITimeoutError, APIStatusError

from interpreter import PersistentInterpreter
from tools import make_tools

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 25
SYSTEM_PROMPT = """You are an expert ML engineer solving a Kaggle competition.
Your goal: produce a valid submission.csv that maximizes the competition score.

## Execution Plan (follow this order strictly)

**Iteration 1:** Call inspect_csv on train.csv AND test.csv (both in one step if possible).
**Iteration 2:** Call read_file on the description/overview file AND inspect sample_submission.csv.
**Iterations 3-4:** Feature engineering + preprocessing pipeline in run_python.
**Iterations 5-9:** Train CatBoost, LightGBM, and XGBoost with cross-validation. Print CV score after each.
**Iterations 10-11:** Build ensemble (weighted average by CV score). Save as submission.csv.
**Iteration 12:** Call validate_submission to confirm the file is correct. Fix any issues.
**Final:** Say "DONE".

## Feature Engineering (always apply for tabular data)
- Parse composite string columns (e.g. "A/12/B" → deck, num, side)
- Create aggregate features: group means/medians, ratios, sums
- Fill missing numerics with median, categoricals with mode
- Label-encode or ordinal-encode all string/object columns
- For binary classification: check class balance, use scale_pos_weight if imbalanced

## Model Training (use these exact defaults)
```python
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import StratifiedKFold, KFold
import numpy as np

# CatBoost — handles categoricals natively, pass cat_features list
cat_model = CatBoostClassifier(
    iterations=1000, learning_rate=0.05, depth=6,
    eval_metric='Accuracy', random_seed=42,
    early_stopping_rounds=50, verbose=100
)

# LightGBM
lgb_model = LGBMClassifier(
    n_estimators=1000, learning_rate=0.05, num_leaves=31,
    random_state=42, n_jobs=-1, verbose=-1
)

# XGBoost
xgb_model = XGBClassifier(
    n_estimators=1000, learning_rate=0.05, max_depth=6,
    random_state=42, n_jobs=-1, verbosity=0,
    eval_metric='logloss', early_stopping_rounds=50
)
```

## Ensemble
- Use 5-fold CV; collect out-of-fold predictions and test predictions per fold
- Weight models by their CV score (higher CV = higher weight)
- Average weighted test predictions for final submission

## Rules
- NEVER read_file on large CSVs — use inspect_csv
- After every run_python, check output for errors before continuing
- If a model throws an error, fix it — do not skip it
- A valid submission is mandatory; fall back to a single model if ensemble fails
- Write submission.csv to WORKDIR

## Important
numpy, pandas, sklearn are pre-imported. Use WORKDIR variable for file paths.
For regression: use RMSE/MAE CV metric. For classification: use accuracy or AUC.
"""


def run_ml_agent(workdir: str, instructions: str, on_status=None) -> str:
    """
    Run the ML agent loop.
    Returns path to submission.csv, or raises an exception.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("MODEL_NAME", "nvidia/nemotron-3-super-120b-a12b:free")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")

    client = OpenAI(api_key=api_key, base_url=base_url)
    interpreter = PersistentInterpreter()
    tool_schemas, dispatch = make_tools(interpreter, workdir)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Competition instructions:\n{instructions}\n\n"
                f"Working directory: {workdir}\n\n"
                "Start by exploring the files, then build and submit your solution."
            ),
        },
    ]

    submission_path = os.path.join(workdir, "submission.csv")

    for iteration in range(MAX_ITERATIONS):
        if on_status:
            on_status(f"Iteration {iteration + 1}/{MAX_ITERATIONS}...")

        logger.info(f"Iteration {iteration + 1}")

        response = None
        for attempt in range(4):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=tool_schemas,
                    tool_choice="auto",
                    max_tokens=4096,
                )
                break
            except (APIConnectionError, APITimeoutError) as api_err:
                if attempt < 3:
                    wait = 2 ** attempt  # 1, 2, 4 seconds
                    logger.warning(f"Connection/timeout error (attempt {attempt+1}/4): {api_err}. Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise
            except APIStatusError as api_err:
                if api_err.status_code in (429, 500, 502, 503) and attempt < 3:
                    wait = 2 ** attempt
                    logger.warning(f"API status {api_err.status_code} (attempt {attempt+1}/4): {api_err}. Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise  # Don't retry 400, 401, 403, or exhausted retries
            except Exception:
                raise  # Don't retry unknown errors

        msg = response.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        # Check if agent is done
        if msg.content and "DONE" in msg.content.upper():
            logger.info("Agent signalled DONE")
            break

        # No tool calls → agent is thinking or finished
        if not msg.tool_calls:
            logger.info("No tool calls, checking for submission...")
            if os.path.exists(submission_path):
                break
            # Prompt agent to continue
            messages.append({
                "role": "user",
                "content": "Continue. If you have not yet created submission.csv, do so now.",
            })
            continue

        # Execute tool calls
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            logger.info(f"Tool call: {name}({list(args.keys())})")
            if on_status:
                on_status(f"Running tool: {name}")

            result = dispatch(name, args)

            # Truncate very long outputs
            if len(result) > 15000:
                result = result[:10000] + "\n...[truncated]...\n" + result[-5000:]

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    interpreter.close()

    # Last resort: if submission.csv missing, try to finalise
    if not os.path.exists(submission_path):
        raise FileNotFoundError(
            f"submission.csv not found in {workdir} after {MAX_ITERATIONS} iterations"
        )

    return submission_path
