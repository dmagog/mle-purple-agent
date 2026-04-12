# MLE Purple Agent

A general-purpose ML engineering agent for the [AgentX-AgentBeats](https://rdi.berkeley.edu/agentx-agentbeats.html) competition (MLE-bench track).

## How it works

1. Receives a Kaggle competition bundle (tar.gz) from the MLE-bench green agent via A2A protocol
2. Extracts competition data and instructions
3. Uses an LLM in an iterative loop with tools (`run_python`, `read_file`, `list_files`, `inspect_csv`) to build an ML solution
4. Returns `submission.csv` as an A2A artifact

## Stack

- **Protocol**: A2A (Google Agent-to-Agent)
- **LLM**: NVIDIA Nemotron 3 Super 120B via OpenRouter (free tier)
- **ML libs**: XGBoost, LightGBM, scikit-learn, pandas
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
