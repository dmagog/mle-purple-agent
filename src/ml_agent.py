"""
LLM loop: iteratively generates and executes Python code to solve an ML competition.
Uses OpenAI-compatible API (works with OpenRouter).
"""
import json
import os
import logging

from openai import OpenAI

from interpreter import PersistentInterpreter
from tools import make_tools

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 15
SYSTEM_PROMPT = """You are an expert ML engineer solving a Kaggle competition.

Your goal: produce a valid submission.csv file that maximizes the competition score.

## Workflow
1. Call list_files to see what's available.
2. Call inspect_csv on train.csv and test.csv to understand the data.
3. Read the task description file (overview.txt, description.md, or similar).
4. Write Python code to build and train a model, then generate predictions.
5. Save predictions as submission.csv matching the sample_submission.csv format exactly.

## Rules
- Use inspect_csv for any CSV file, never read_file on large data files.
- After each run_python call, check the output carefully for errors.
- Always validate submission.csv format against sample_submission.csv before finishing.
- Prefer fast, reliable models: XGBoost, LightGBM, RandomForest, or simple neural nets.
- For tabular data: do feature engineering, handle missing values, use cross-validation.
- If a model fails, try a simpler approach — a valid submission beats no submission.
- When submission.csv is ready and validated, say "DONE" and stop calling tools.

## Important
All files are in WORKDIR (set automatically). Use relative or absolute paths.
Write submission.csv to WORKDIR.
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

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tool_schemas,
            tool_choice="auto",
            max_tokens=4096,
        )

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
            if len(result) > 8000:
                result = result[:4000] + "\n...[truncated]...\n" + result[-2000:]

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
