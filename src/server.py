"""
A2A server entry point.
"""
import argparse
import logging
import uvicorn
from dotenv import load_dotenv

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from executor import Executor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="MLE Purple Agent A2A server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--card-url", type=str, default=None)
    args = parser.parse_args()

    skill = AgentSkill(
        id="ml_competition_solver",
        name="ML Competition Solver",
        description=(
            "Receives a Kaggle competition bundle (tar.gz) with data and instructions. "
            "Analyses the data, trains an ML model, and returns submission.csv."
        ),
        tags=["mle-bench", "kaggle", "machine-learning", "tabular"],
        examples=[],
    )

    agent_card = AgentCard(
        name="MLE Purple Agent",
        description=(
            "A general-purpose ML engineering agent that solves Kaggle-style competitions. "
            "Supports tabular, CV, NLP tasks. Returns a valid submission.csv."
        ),
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version="1.0.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )

    request_handler = DefaultRequestHandler(
        agent_executor=Executor(),
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
        max_content_length=None,
    )

    uvicorn.run(server.build(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
