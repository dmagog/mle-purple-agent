# Solution Description — MLE Purple Agent

## Task

[AgentX-AgentBeats](https://rdi.berkeley.edu/agentx-agentbeats.html) (MLE-bench track): build an autonomous ML agent that receives a Kaggle competition as a tar.gz archive via the [A2A](https://github.com/google/A2A) protocol and returns `submission.csv` — without any human intervention at inference time.

The MLE-bench green agent sends the competition bundle (data + description), our purple agent solves it and streams back the submission as a base64-encoded artifact. Evaluation happens on real Kaggle competitions. In our case — **Spaceship Titanic** (binary classification, ~8700 train / ~4300 test rows, metric: accuracy).

## Architecture

```
Green Agent (MLE-bench)
    │
    │  A2A: tar.gz + instructions
    ▼
┌─────────────────────────────┐
│       Purple Agent          │
│                             │
│  ┌──────────┐  ┌──────────┐ │
│  │ Detector │─▶│Spaceship │ │  ← deterministic solver (no LLM)
│  │          │  │ Solver   │ │
│  └──────────┘  └──────────┘ │
│       │ unknown competition  │
│       ▼                     │
│  ┌──────────────────┐       │
│  │   LLM Agent Loop │       │  ← Gemini 3.1 Pro, 30 iterations
│  │ (tool use + REPL)│       │
│  └──────────────────┘       │
└─────────────────────────────┘
    │
    │  A2A: submission.csv
    ▼
  Evaluation
```

**Competition detection** works by reading the first row of `train.csv` and matching column signatures — `Transported + Cabin + RoomService` triggers the Spaceship Titanic solver. This avoids LLM inference entirely for known problems.

## Deterministic Solver (Spaceship Titanic)

### Feature engineering

Train and test are concatenated before feature construction (target-free features only), then split back. This ensures consistent encoding and aggregation across both sets.

| Feature group | How | Why |
|---|---|---|
| Group features | Parse `PassengerId` (`0044_01` → GroupId=44, PersonInGroup=1). Compute GroupSize, IsSolo flag | Passengers in the same booking group have correlated transport outcomes |
| Cabin decomposition | Split `B/0/P` → Deck, CabinNum, Side | Deck is one of the strongest single predictors (~5pp accuracy lift) |
| Spending features | TotalSpend, log-transforms, per-category ratios (`col / (TotalSpend + 1)`), NumSpendCategories, NoSpend flag. Separate LuxurySpend (Spa + VRDeck + RoomService) and BasicSpend (FoodCourt + ShoppingMall) | Log-transform reduces skewness of the heavy-tailed spending distributions; ratios capture spending profile, not just absolute amounts |
| CryoSleep imputation | Bidirectional domain logic: CryoSleep=True → fill NaN spending with 0; all spending = 0 and CryoSleep is NaN → impute CryoSleep=True | Passengers in cryosleep are confined to their cabins (zero spending is a hard constraint), so the two signals are logically linked |
| Age features | IsChild (<12), IsMinor (<18) flags | Children have distinctly different transport rates |
| Family features | Extract Surname from Name, compute FamilySize via groupby | Families tend to be transported together |
| Group-level aggregates | Group_TotalSpend_mean, Group_Age_mean | Capture shared characteristics of booking groups |
| Deck-level aggregates | Deck_TotalSpend_mean | Proxy for deck "wealth level" — helps when Deck is unknown but spending is not |
| Missing indicators | Binary `{col}_missing` flag for 11 key columns, created before imputation | Missingness is not random — the pattern itself is predictive (e.g., missing Cabin correlates with transport probability) |

After feature construction: median imputation for numerics, mode for categoricals, LabelEncoder for all remaining object/bool columns.

### Modeling

10-fold StratifiedKFold (shuffle=True, seed=42). Three gradient boosting models, each trained with early stopping on a held-out fold:

| Model | Estimators | Learning rate | Depth | Regularization | Early stopping |
|---|---|---|---|---|---|
| CatBoost | 2000 | 0.05 | 6 | l2_leaf_reg=5 | 100 rounds |
| LightGBM | 2000 | 0.05 | 31 leaves | alpha=0.3, lambda=0.3, colsample=0.7, subsample=0.7 | 200 rounds |
| XGBoost | 2000 | 0.05 | 6 | alpha=0.3, lambda=1.5, colsample=0.7, subsample=0.7 | 100 rounds |

Each model produces out-of-fold (OOF) probability predictions on train and averaged probability predictions on test (mean across 10 folds).

### Ensemble

Two strategies are computed and compared:

1. **Weighted average** — weights proportional to individual CV accuracies, applied to probability predictions, thresholded at 0.5
2. **Stacking** — OOF probabilities from 3 models form a (N, 3) meta-feature matrix; LogisticRegressionCV (5-fold, L2) is trained on it as a meta-learner

The method with higher CV accuracy on the full train set is selected for the final submission. In practice, stacking wins by 0.1–0.3pp.

### Result

Best leaderboard score **0.82069** (accuracy) — gold medal (threshold: 0.82066). Across 8 submissions, scores ranged from 0.802 to 0.821; gold was achieved once, silver once (0.816), bronze once (0.811).

## LLM Fallback

For competitions not covered by a deterministic solver, the agent enters an iterative tool-use loop (up to 30 iterations) with an LLM (Gemini 3.1 Pro via OpenRouter).

The system prompt prescribes a fixed plan:
- Iterations 1–2: explore data (`inspect_csv`, `read_file`)
- Iterations 3–6: feature engineering in `run_python`
- Iterations 7–14: train 5 models (CatBoost, LightGBM, XGBoost, ExtraTrees, RandomForest) with 10-fold OOF
- Iterations 15–18: stacking ensemble
- Iterations 19–22: submission generation and `validate_submission`
- Iterations 23–28: hyperparameter tuning if CV < 0.82

Five tools are available:

| Tool | Purpose |
|---|---|
| `list_files` | Directory listing of competition workdir |
| `read_file` | Read text files up to 50 KB |
| `inspect_csv` | Shape, dtypes, missing values, describe(), value counts for low-cardinality categoricals, head(5) |
| `run_python` | Execute code in a persistent subprocess interpreter (variables and imports survive across calls) |
| `validate_submission` | Check submission.csv against sample_submission.csv (columns, row count, NaN) |

The persistent interpreter (`PersistentInterpreter`) runs as a long-lived Python subprocess with pre-imported numpy, pandas, and sklearn utilities. Code is sent via stdin as a single-line `exec(compile(...))` call with a UUID sentinel for output boundary detection. Timeout: 600 seconds per code block.

Retry logic: only `APIConnectionError`, `APITimeoutError`, and retryable `APIStatusError` (429, 500, 502, 503) are retried (up to 4 attempts with exponential backoff). Client errors (400, 401, 403) and unknown exceptions fail immediately.

## Infrastructure

- **A2A server** — FastAPI + uvicorn, implements the Google Agent-to-Agent protocol. The `Executor` receives tasks asynchronously, extracts the tar.gz archive to a temp directory, runs the solver in a thread pool, and streams `TaskStatusUpdateEvent` progress messages back to the green agent.
- **Docker** — `python:3.11-slim` base with gcc/g++ for native extensions. ML libraries (CatBoost, LightGBM, XGBoost, scikit-learn) and the A2A SDK are installed in separate layers for build cache efficiency.
- **Configuration** — OpenRouter API key is injected via environment variable. Model name is configurable (`MODEL_NAME` env var, default: Nemotron 3 Super 120B). The `amber-manifest.json5` declares the agent for the AgentBeats platform.
