"""
Tools available to the ML agent LLM.
"""
import os
import json


def make_tools(interpreter, workdir: str) -> list[dict]:
    """Return OpenAI-style tool definitions + a dispatcher."""

    schemas = [
        {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List all files in the competition working directory recursively.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a text file. Do NOT use for large CSVs — use inspect_csv instead.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Relative path inside workdir"}
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "inspect_csv",
                "description": (
                    "Show shape, dtypes, missing values, and first 5 rows of a CSV file. "
                    "Always use this instead of read_file for CSV/TSV data files."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Relative path inside workdir"}
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_python",
                "description": (
                    "Execute Python code in a persistent interpreter. "
                    "Variables and imports persist between calls. "
                    "All file paths must be absolute or relative to the working directory. "
                    "Print results you want to see. "
                    "Write submission.csv to the workdir when ready."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Python code to execute"}
                    },
                    "required": ["code"],
                },
            },
        },
    ]

    def dispatch(name: str, args: dict) -> str:
        if name == "list_files":
            return _list_files(workdir)
        elif name == "read_file":
            return _read_file(workdir, args["path"])
        elif name == "inspect_csv":
            return _inspect_csv(interpreter, workdir, args["path"])
        elif name == "run_python":
            # Inject workdir as WORKDIR variable on first use
            setup = f"import os; os.chdir({repr(workdir)}); WORKDIR = {repr(workdir)}\n"
            full_code = setup + args["code"]
            return interpreter.run(full_code)
        else:
            return f"Unknown tool: {name}"

    return schemas, dispatch


def _list_files(workdir: str) -> str:
    result = []
    for root, dirs, files in os.walk(workdir):
        # Skip hidden dirs
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for f in files:
            full = os.path.join(root, f)
            rel = os.path.relpath(full, workdir)
            size = os.path.getsize(full)
            result.append(f"{rel}  ({size:,} bytes)")
    return "\n".join(result) if result else "(empty)"


def _read_file(workdir: str, path: str) -> str:
    full = os.path.join(workdir, path)
    if not os.path.exists(full):
        return f"File not found: {path}"
    size = os.path.getsize(full)
    if size > 50_000:
        return f"File too large ({size:,} bytes). Use inspect_csv for data files."
    try:
        with open(full, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"


def _inspect_csv(interpreter, workdir: str, path: str) -> str:
    full = os.path.join(workdir, path)
    if not os.path.exists(full):
        return f"File not found: {path}"
    code = f"""
import pandas as pd
df = pd.read_csv({repr(full)})
print(f"Shape: {{df.shape}}")
print(f"Columns: {{list(df.columns)}}")
print(f"Dtypes:\\n{{df.dtypes}}")
print(f"Missing values:\\n{{df.isnull().sum()}}")
print(f"\\nFirst 5 rows:\\n{{df.head()}}")
"""
    return interpreter.run(code)
