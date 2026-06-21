# MLE Purple Agent

*An autonomous ML-engineering agent built for [AgentX–AgentBeats](https://rdi.berkeley.edu/agentx-agentbeats.html) — a competition where the benchmarks are themselves agents, and agents are graded by agents.*

<div align="center">
<table>
  <tr>
    <td align="center">
      <h3>🥇 Cleared MLE-bench's gold bar — autonomously</h3>
      On the <b>Spaceship Titanic</b> task the agent scored <b>0.82069</b>, just past MLE-bench's <b>0.82066</b> gold threshold &mdash; produced end-to-end, no human in the loop.<br/>
      <sub>For scale, 0.82 is around the top 6% of the current Spaceship Titanic leaderboard &mdash;<br/> genuinely strong, though the headline is the autonomy, not the rank.</sub><br/><br/>
      <a href="https://agentbeats.dev/agentbeater/mle-bench">Leaderboard</a> ·
      <a href="https://github.com/RDI-Foundation/MLE-bench-agentbeats-leaderboard/commit/c181aee126e02548e61e3bbef3bce1f8ea9f49e1">Result</a> ·
      <a href="SOLUTION.md">Solution write-up</a>
    </td>
  </tr>
</table>
</div>

## Competition

[AgentX-AgentBeats](https://rdi.berkeley.edu/agentx-agentbeats.html) is an international competition in **building AI agents**, organized by [Berkeley RDI](https://rdi.berkeley.edu/) alongside the Agentic AI MOOC (~40 000 registered learners). 1 300+ teams from 100+ countries, prize pool over $1M in cash, cloud credits and API resources from sponsors including DeepMind, OpenAI, AWS, Meta, Snowflake, Hugging Face, Lambda and Sierra.

Its defining idea: the benchmarks aren't fixed scripts — they're *agents* too. In **Phase 1** teams build *green agents* — evaluator agents that set and grade real-world tasks. In **Phase 2** teams build *purple agents* — the competitors that solve them. Agents grading agents. Tracks cover ML engineering, coding, finance, healthcare, cybersecurity, multi-agent evaluation and more.

**MLE-bench** is an ML engineering track based on [OpenAI's MLE-bench](https://arxiv.org/abs/2410.07095) — a suite of real Kaggle competitions used to evaluate whether AI agents can do end-to-end machine learning: read data, engineer features, train models and produce a valid submission, all without human intervention. The green agent sends the competition as a tar.gz archive via the [A2A protocol](https://github.com/google/A2A); the purple agent must return `submission.csv`.

## How it works

1. Receives a Kaggle competition bundle (tar.gz) from the MLE-bench green agent via A2A
2. Detects known competitions (e.g. Spaceship Titanic) and runs a deterministic solver with optimized feature engineering and ensemble
3. For unknown competitions — uses an LLM in an iterative loop with tools (`run_python`, `read_file`, `list_files`, `inspect_csv`, `validate_submission`)
4. Returns `submission.csv` as an A2A artifact

## Provenance

The gold result (**0.82069**, 13 Apr 2026 — [run record](https://github.com/RDI-Foundation/MLE-bench-agentbeats-leaderboard/commit/c181aee126e02548e61e3bbef3bce1f8ea9f49e1)) was produced by the **LLM-agent loop** (`src/ml_agent.py`), which improvises an ML pipeline per task; across the agent's eight runs scores ranged 0.802–0.821, with gold reached once. The deterministic `src/solve_spaceship.py` is a cleaner, reproducible reimplementation of that approach, added *after* the competition as a fast path for the known task. The exact gold-winning build is preserved as git tag `gold-2026-04-13` and Docker image `dmagog/mle-purple-agent@sha256:97d33c…`.

## Notebook

A step-by-step educational walkthrough of the Spaceship Titanic solver (EDA → feature engineering → 3-GBDT stacking ensemble) is published on Kaggle and included in this repo:

- **Kaggle** (rendered, runnable): [Agents Grading Agents: Spaceship Titanic MLE-bench](https://www.kaggle.com/code/georgymamarin/agents-grading-agents-spaceship-titanic-mle-bench)
- **In this repo**: [`spaceship-titanic-gold-medal-guide.ipynb`](spaceship-titanic-gold-medal-guide.ipynb)

## Stack

- **Protocol**: A2A (Google Agent-to-Agent)
- **LLM fallback**: Gemini 2.5 Pro via OpenRouter
- **ML libs**: CatBoost, LightGBM, XGBoost, scikit-learn, pandas
- **Ensemble**: 10-fold stacking (CatBoost + LightGBM + XGBoost) with LogisticRegressionCV meta-learner
- **Server**: FastAPI + uvicorn

## Docker

```bash
docker pull dmagog/mle-purple-agent:latest
docker run -e OPENROUTER_API_KEY=your_key -p 8000:8000 dmagog/mle-purple-agent:latest
```

## Agent card

```
GET http://localhost:8000/.well-known/agent-card.json
```
