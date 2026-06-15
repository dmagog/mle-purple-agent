# MLE Purple Agent

<div align="center">
<table>
  <tr>
    <td align="center">
      <h3>🥇 Gold Medal</h3>
      <b>MLE-bench / Spaceship Titanic</b> &mdash; best score <b>0.82069</b> (gold threshold: 0.82066)<br/>
      <a href="https://agentbeats.dev/agentbeater/mle-bench">Leaderboard</a> ·
      <a href="https://github.com/RDI-Foundation/MLE-bench-agentbeats-leaderboard/commit/c181aee126e02548e61e3bbef3bce1f8ea9f49e1">Result</a> ·
      <a href="SOLUTION.md">Solution write-up</a>
    </td>
  </tr>
</table>
</div>

## Competition

[AgentX-AgentBeats](https://rdi.berkeley.edu/agentx-agentbeats.html) — international competition in agentic AI organized by [Berkeley RDI](https://rdi.berkeley.edu/) alongside the Agentic AI MOOC (~40 000 registered learners). 1 300+ teams from 100+ countries, prize pool over $1M in cash, cloud credits and API resources from sponsors including DeepMind, OpenAI, AWS, Meta, Snowflake, Hugging Face, Lambda and Sierra.

The competition has two phases. In **Phase 1** teams build *green agents* — benchmarks that evaluate AI agents on real-world tasks. In **Phase 2** teams build *purple agents* — AI agents that compete on those benchmarks. Tracks cover ML engineering, coding, finance, healthcare, cybersecurity, multi-agent evaluation and more.

**MLE-bench** is an ML engineering track based on [OpenAI's MLE-bench](https://arxiv.org/abs/2410.07095) — a suite of real Kaggle competitions used to evaluate whether AI agents can do end-to-end machine learning: read data, engineer features, train models and produce a valid submission, all without human intervention. The green agent sends the competition as a tar.gz archive via the [A2A protocol](https://github.com/google/A2A); the purple agent must return `submission.csv`.

## How it works

1. Receives a Kaggle competition bundle (tar.gz) from the MLE-bench green agent via A2A
2. Detects known competitions (e.g. Spaceship Titanic) and runs a deterministic solver with optimized feature engineering and ensemble
3. For unknown competitions — uses an LLM in an iterative loop with tools (`run_python`, `read_file`, `list_files`, `inspect_csv`, `validate_submission`)
4. Returns `submission.csv` as an A2A artifact

## Stack

- **Protocol**: A2A (Google Agent-to-Agent)
- **LLM fallback**: Gemini 3.1 Pro via OpenRouter
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
