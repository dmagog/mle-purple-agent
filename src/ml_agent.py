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

MAX_ITERATIONS = 30
SYSTEM_PROMPT = """You are a top-ranked Kaggle Grandmaster solving a competition.
Your goal: produce submission.csv that achieves a GOLD medal score (top 10%).

## Execution Plan (follow this order strictly)

**Phase 1 — Explore (iterations 1-2):**
- inspect_csv on train.csv, test.csv, sample_submission.csv
- read_file on any description/overview file

**Phase 2 — Feature Engineering (iterations 3-6):**

This is the MOST IMPORTANT phase. Build ALL of these features in run_python:

### Domain-Specific Features (Spaceship Titanic)
```python
# 1. PassengerId → group features
df['GroupId'] = df['PassengerId'].apply(lambda x: x.split('_')[0]).astype(int)
df['PersonInGroup'] = df['PassengerId'].apply(lambda x: x.split('_')[1]).astype(int)
group_sizes = df.groupby('GroupId')['PersonInGroup'].transform('count')
df['GroupSize'] = group_sizes
df['IsSolo'] = (df['GroupSize'] == 1).astype(int)

# 2. Cabin → deck, num, side
df['Deck'] = df['Cabin'].apply(lambda x: x.split('/')[0] if pd.notna(x) else 'Unknown')
df['CabinNum'] = df['Cabin'].apply(lambda x: int(x.split('/')[1]) if pd.notna(x) else -1)
df['Side'] = df['Cabin'].apply(lambda x: x.split('/')[2] if pd.notna(x) else 'Unknown')

# 3. Spending features (RoomService, FoodCourt, ShoppingMall, Spa, VRDeck)
spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
df['TotalSpend'] = df[spend_cols].sum(axis=1)
df['LogTotalSpend'] = np.log1p(df['TotalSpend'])
df['NoSpend'] = (df['TotalSpend'] == 0).astype(int)
df['NumSpendCategories'] = (df[spend_cols] > 0).sum(axis=1)
for col in spend_cols:
    df[f'{col}_ratio'] = df[col] / (df['TotalSpend'] + 1)
    df[f'Log{col}'] = np.log1p(df[col])
# Luxury vs necessity
df['LuxurySpend'] = df['Spa'] + df['VRDeck'] + df['RoomService']
df['BasicSpend'] = df['FoodCourt'] + df['ShoppingMall']

# 4. CryoSleep imputation: if CryoSleep=True, all spending must be 0
# Use this BEFORE filling missing values
cryo_mask = df['CryoSleep'] == True
for col in spend_cols:
    df.loc[cryo_mask, col] = df.loc[cryo_mask, col].fillna(0)
# Reverse: if all spending is 0, likely CryoSleep=True
no_spend_mask = (df[spend_cols].sum(axis=1) == 0) & df['CryoSleep'].isna()
df.loc[no_spend_mask, 'CryoSleep'] = True

# 5. Age features
df['AgeBin'] = pd.cut(df['Age'], bins=[0,12,18,25,35,50,65,200],
                       labels=['Child','Teen','YoungAdult','Adult','MidAge','Senior','Elder'])
df['IsChild'] = (df['Age'] < 12).astype(int)
df['IsMinor'] = (df['Age'] < 18).astype(int)

# 6. Name → surname → family size
df['Surname'] = df['Name'].apply(lambda x: x.split()[-1] if pd.notna(x) else 'Unknown')
surname_counts = df.groupby('Surname')['Name'].transform('count')
df['FamilySize'] = surname_counts

# 7. Group-level aggregates
for col in spend_cols + ['Age']:
    df[f'Group_{col}_mean'] = df.groupby('GroupId')[col].transform('mean')
    df[f'Group_{col}_std'] = df.groupby('GroupId')[col].transform('std').fillna(0)

# 8. Deck-level aggregates
df['Deck_TotalSpend_mean'] = df.groupby('Deck')['TotalSpend'].transform('mean')

# 9. Missing value indicators BEFORE imputation
for col in df.columns:
    if df[col].isnull().any():
        df[f'{col}_missing'] = df[col].isnull().astype(int)
```

### General preprocessing
- Fill missing numerics with median, categoricals with mode
- Label-encode ALL remaining object/category columns
- Drop PassengerId, Name, Cabin, Surname (already extracted features)
- Print final shape and feature list

**Phase 3 — Train 5 Models with OOF (iterations 7-14):**

Use 10-fold StratifiedKFold. For EACH model:
- Collect out-of-fold (OOF) predictions on train
- Collect averaged test predictions across folds
- Print CV accuracy

```python
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Model 1: CatBoost
CatBoostClassifier(iterations=3000, learning_rate=0.03, depth=6,
    l2_leaf_reg=3, random_seed=42, early_stopping_rounds=200, verbose=0)

# Model 2: LightGBM
LGBMClassifier(n_estimators=3000, learning_rate=0.03, num_leaves=31,
    min_child_samples=20, reg_alpha=0.1, reg_lambda=0.1,
    colsample_bytree=0.8, subsample=0.8, subsample_freq=5,
    random_state=42, n_jobs=-1, verbose=-1)

# Model 3: XGBoost
XGBClassifier(n_estimators=3000, learning_rate=0.03, max_depth=6,
    min_child_weight=3, reg_alpha=0.1, reg_lambda=1.0,
    colsample_bytree=0.8, subsample=0.8,
    random_state=42, n_jobs=-1, verbosity=0,
    eval_metric='logloss', early_stopping_rounds=200)

# Model 4: ExtraTrees
ExtraTreesClassifier(n_estimators=2000, max_depth=None,
    min_samples_leaf=2, random_state=42, n_jobs=-1)

# Model 5: RandomForest
RandomForestClassifier(n_estimators=2000, max_depth=None,
    min_samples_leaf=2, random_state=42, n_jobs=-1)
```

Store ALL OOF predictions in oof_preds dict and test predictions in test_preds dict.

**Phase 4 — Stacking Ensemble (iterations 15-18):**

```python
from sklearn.linear_model import LogisticRegressionCV
import numpy as np

# Stack OOF predictions as meta-features
oof_stack = np.column_stack([oof_preds[m] for m in model_names])
test_stack = np.column_stack([test_preds[m] for m in model_names])

# Meta-learner
meta = LogisticRegressionCV(cv=5, random_state=42, max_iter=1000)
meta.fit(oof_stack, y_train)
meta_cv_score = meta.score(oof_stack, y_train)
print(f"Stacking CV accuracy: {meta_cv_score:.5f}")

# Also try simple weighted average
# Weight by individual CV scores
weights = np.array([cv_scores[m] for m in model_names])
weights = weights / weights.sum()
weighted_oof = (oof_stack * weights).sum(axis=1)
weighted_cv = accuracy_score(y_train, (weighted_oof > 0.5).astype(int))
print(f"Weighted average CV accuracy: {weighted_cv:.5f}")

# Pick the better ensemble method
```

**Phase 5 — Generate submission & validate (iterations 19-22):**
- Use the best ensemble to predict on test
- For classification: threshold=0.5, convert to True/False
- Write submission.csv matching sample_submission.csv format exactly
- Call validate_submission

**Phase 6 — Manual hyperparameter tuning (iterations 23-28):**
If CV score < 0.82, tune the best model manually:
- Try 3 variations of learning_rate: [0.01, 0.03, 0.05]
- Try 3 variations of depth: [4, 6, 8]
- Try 3 variations of regularization: [0.01, 0.1, 1.0]
If any variation improves CV, rebuild ensemble with tuned model and regenerate submission.csv.

**Iteration 29-30:** Say "DONE"

## Critical Rules
- NEVER read_file on large CSVs — use inspect_csv only
- After EVERY run_python, check output for errors before continuing
- ALWAYS print CV scores — never train blind
- If a model errors, FIX it immediately
- DO NOT write submission.csv until you have trained ALL 5 models and built the ensemble
- submission.csv MUST match sample_submission.csv format exactly
- Write submission.csv to WORKDIR
- numpy, pandas, sklearn are pre-imported. Use WORKDIR variable for file paths.
"""


def _is_spaceship_titanic(workdir: str, instructions: str) -> bool:
    """Detect if this is the Spaceship Titanic competition."""
    # Check instructions text
    text = instructions.lower()
    if "spaceship" in text and "titanic" in text:
        return True
    # Check if train.csv has the Transported column
    train_path = os.path.join(workdir, "train.csv")
    if os.path.exists(train_path):
        try:
            import pandas as pd
            cols = list(pd.read_csv(train_path, nrows=0).columns)
            if "Transported" in cols and "Cabin" in cols and "RoomService" in cols:
                return True
        except Exception:
            pass
    return False


def run_ml_agent(workdir: str, instructions: str, on_status=None) -> str:
    """
    Run the ML agent loop.
    Returns path to submission.csv, or raises an exception.
    """
    # Try deterministic solver for known competitions
    if _is_spaceship_titanic(workdir, instructions):
        logger.info("Detected Spaceship Titanic — using deterministic solver")
        if on_status:
            on_status("Detected Spaceship Titanic — using optimized solver")
        try:
            from solve_spaceship import solve
            return solve(workdir, on_status=on_status)
        except Exception as e:
            logger.exception(f"Deterministic solver failed: {e}, falling back to LLM")
            if on_status:
                on_status(f"Optimized solver failed, falling back to LLM agent...")

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
