# Plan: Purple Agent Improvements for MLE-bench

## Current State

Agent achieves ~0.75-0.79 on spaceship-titanic. Top leaderboard scores: 0.82-0.83.
Architecture is solid (persistent interpreter, A2A integration, tool-based LLM loop),
but several concrete issues limit the score and reliability.

---

## BUG FIX (Critical)

### B1: CatBoost missing from Docker image

**File:** `Dockerfile`
**Problem:** `catboost>=1.2.0` is in `requirements.txt` but NOT installed in the Dockerfile.
The agent has access to XGBoost and LightGBM, but CatBoost (which handles categorical
features natively and often wins on tabular data) is unavailable at runtime.
**Fix:** Add `catboost>=1.2.0` to the pip install layer in Dockerfile.

---

## HIGH PRIORITY (Direct score impact)

### H1: Rewrite system prompt with concrete strategy

**File:** `src/ml_agent.py` — `SYSTEM_PROMPT`
**Problem:** Current prompt is generic ("explore files, build model"). The agent wastes
3-5 iterations on exploration (list_files, inspect_csv, read description) before writing
any code. With only 15 iterations, this leaves ~10 for actual modeling.
**Fix:** Replace with a directive prompt that prescribes the exact approach:

- Iteration 1: `inspect_csv` on train.csv + test.csv (one call each)
- Iteration 2: `read_file` on description + sample_submission
- Iteration 3-4: Feature engineering + preprocessing pipeline
- Iteration 5-8: Train CatBoost + LightGBM + XGBoost
- Iteration 9-10: Ensemble (weighted average or stacking)
- Iteration 11: Validate submission format + save
- Include specific feature engineering hints for tabular data:
  - Parse composite columns (e.g., Cabin -> deck/num/side)
  - Create aggregate features (sums, ratios)
  - Handle missing values with median/mode per group
  - Label/ordinal encode categoricals
- Tell agent to print CV score after each model

### H2: Increase MAX_ITERATIONS from 15 to 25

**File:** `src/ml_agent.py` — `MAX_ITERATIONS`
**Problem:** 15 iterations is tight. Agent often runs out before trying alternatives
or building an ensemble. Successful competitors use more iterations.
**Fix:** Change `MAX_ITERATIONS = 25`. Cost is minimal (free model), but gives agent
room for error recovery and ensemble building.

### H3: Increase code execution timeout from 120s to 300s

**File:** `src/interpreter.py` — `TIMEOUT`
**Problem:** CatBoost and XGBoost with cross-validation on moderate datasets can take
2-4 minutes. Current 120s limit kills training mid-run.
**Fix:** Change `TIMEOUT = 300`.

### H4: Selective retry logic (only network errors)

**File:** `src/ml_agent.py` — retry loop
**Problem:** Current code catches all `Exception`, including JSON decode errors, auth
errors, and invalid request errors. These won't resolve with retry and waste time.
**Fix:** Catch only retryable errors:
```python
from openai import APIStatusError, APIConnectionError, APITimeoutError

for attempt in range(4):
    try:
        response = client.chat.completions.create(...)
        break
    except (APIConnectionError, APITimeoutError) as e:
        # Always retry connection/timeout errors
        ...
    except APIStatusError as e:
        if e.status_code in (429, 500, 502, 503):
            # Retry rate limits and server errors
            ...
        else:
            raise  # Don't retry 400, 401, 403, etc.
    except Exception:
        raise  # Don't retry unknown errors
```

### H5: Add ensemble strategy to system prompt

**File:** `src/ml_agent.py` — `SYSTEM_PROMPT`
**Problem:** Agent typically trains one model and stops. Top scores require ensembling
2-3 models (CatBoost + LightGBM + XGBoost) with weighted averaging.
**Fix:** Explicitly instruct agent in the system prompt:
- "Train at least 2 models with cross-validation"
- "Average their predictions for the final submission"
- "Weight models by their CV score"

---

## MEDIUM PRIORITY (Robustness)

### M1: Add submission validation tool

**File:** `src/tools.py`
**Problem:** Agent sometimes generates malformed submission.csv (wrong columns, wrong
number of rows, wrong dtypes). This is caught only at evaluation time.
**Fix:** Add a `validate_submission` tool that:
- Compares submission.csv columns against sample_submission.csv
- Checks row count matches test.csv
- Verifies no NaN values in predictions
- Returns clear error messages if mismatched

### M2: Increase output truncation limit

**File:** `src/ml_agent.py` — truncation block
**Problem:** Outputs >8000 chars are truncated. Training logs, error tracebacks, and
data summaries often exceed this. Agent loses critical information.
**Fix:** Increase to 15000 chars (or 10000+5000 split). Free models typically support
128K+ context, so larger tool outputs are fine.

### M3: Smarter inspect_csv with statistics

**File:** `src/tools.py` — `_inspect_csv`
**Problem:** Current inspect shows only shape, dtypes, nulls, head(5). Missing:
- Value distributions for categoricals (nunique, top values)
- Numeric statistics (mean, std, min, max)
- Target variable distribution
**Fix:** Add `df.describe(include='all')` and `df[col].value_counts().head(5)` for
low-cardinality categoricals.

### M4: Interpreter bootstrap with common imports

**File:** `src/interpreter.py` — `_REPL_BOOTSTRAP`
**Problem:** Agent must import pandas, numpy, sklearn in every run_python call,
wasting tokens and iterations.
**Fix:** Pre-import common libraries in the REPL bootstrap:
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
```

---

## LOW PRIORITY (Nice-to-have)

### L1: Add partial file reading

**File:** `src/tools.py` — `_read_file`
**Problem:** 50KB hard limit. Some description files are larger.
**Fix:** Add optional `max_lines` parameter to read first N lines.

### L2: Model warm-start hints

**File:** `src/ml_agent.py` — `SYSTEM_PROMPT`
**Problem:** Agent doesn't know good default hyperparameters.
**Fix:** Include common good defaults in prompt:
- LightGBM: n_estimators=1000, learning_rate=0.05, early_stopping_rounds=50
- XGBoost: same pattern
- CatBoost: iterations=1000, learning_rate=0.05, cat_features=auto

### L3: Garbage collection between iterations

**File:** `src/interpreter.py`
**Problem:** Large DataFrames accumulate in memory.
**Fix:** Add periodic `gc.collect()` in interpreter between code blocks.

---

## Implementation Order

**Phase 1 — Quick wins (30 min, expect 0.79-0.81):**
1. B1: Add CatBoost to Dockerfile
2. H2: MAX_ITERATIONS = 25
3. H3: TIMEOUT = 300
4. H4: Selective retry

**Phase 2 — Score boost (1 hour, expect 0.81-0.82):**
5. H1: Rewrite system prompt
6. H5: Ensemble strategy in prompt
7. M2: Increase truncation limit
8. M4: Bootstrap common imports

**Phase 3 — Polish (1 hour, expect 0.82+):**
9. M1: Submission validation tool
10. M3: Smarter inspect_csv
11. L2: Model hyperparameter hints

**After each phase:** rebuild Docker, push, and Quick Submit to verify improvement.

---

## Build & Deploy Checklist

```bash
# Build for linux/amd64 (required for GitHub Actions)
docker buildx build --platform linux/amd64 -t dmagog/mle-purple-agent:latest --push .

# Verify image
docker run --rm dmagog/mle-purple-agent:latest python -c "import catboost; print('OK')"

# Quick Submit with:
# - Green secrets: KAGGLE_USERNAME=georgymamarin, KAGGLE_KEY=<legacy_key>
# - Participant secret: openrouter_api_key=<key>
# - All values typed manually (not via form_input automation)
```
