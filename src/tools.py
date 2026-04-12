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
                    "Show shape, dtypes, missing values, statistics, and sample rows of a CSV file. "
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
                    "numpy (np), pandas (pd), and common sklearn utilities are pre-imported. "
                    "WORKDIR variable is set to the competition directory. "
                    "Print results you want to see. "
                    "Write submission.csv to WORKDIR when ready."
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
        {
            "type": "function",
            "function": {
                "name": "validate_submission",
                "description": (
                    "Validate submission.csv against sample_submission.csv. "
                    "Checks columns, row count, dtypes, and missing values. "
                    "Call this before declaring DONE."
                ),
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
    ]

    # One-time setup injected into interpreter
    _setup_done = [False]

    def dispatch(name: str, args: dict) -> str:
        if name == "list_files":
            return _list_files(workdir)
        elif name == "read_file":
            return _read_file(workdir, args["path"])
        elif name == "inspect_csv":
            return _inspect_csv(interpreter, workdir, args["path"])
        elif name == "run_python":
            # Inject workdir as WORKDIR variable on first use only
            if not _setup_done[0]:
                setup = f"import os; os.chdir({repr(workdir)}); WORKDIR = {repr(workdir)}\n"
                interpreter.run(setup)
                _setup_done[0] = True
            return interpreter.run(args["code"])
        elif name == "validate_submission":
            return _validate_submission(workdir)
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
print(f"\\nDtypes:\\n{{df.dtypes}}")
print(f"\\nMissing values:\\n{{df.isnull().sum()}}")
print(f"\\nNumeric statistics:\\n{{df.describe()}}")
# Show value counts for low-cardinality categorical columns
cat_cols = [c for c in df.select_dtypes(include='object').columns if df[c].nunique() <= 20]
if cat_cols:
    print("\\nCategorical value counts:")
    for col in cat_cols[:5]:
        print(f"  {{col}} (nunique={{df[col].nunique()}}): {{df[col].value_counts().head(5).to_dict()}}")
print(f"\\nFirst 5 rows:\\n{{df.head()}}")
"""
    return interpreter.run(code)


def _validate_submission(workdir: str) -> str:
    submission_path = os.path.join(workdir, "submission.csv")
    if not os.path.exists(submission_path):
        return "ERROR: submission.csv not found in workdir."

    # Find sample_submission
    sample_path = None
    for name in ["sample_submission.csv", "sample_submission.csv.gz"]:
        candidate = os.path.join(workdir, name)
        if os.path.exists(candidate):
            sample_path = candidate
            break

    try:
        import pandas as pd
        sub = pd.read_csv(submission_path)

        errors = []
        warnings = []

        if sample_path:
            sample = pd.read_csv(sample_path)
            # Check columns
            if list(sub.columns) != list(sample.columns):
                errors.append(
                    f"Column mismatch: got {list(sub.columns)}, expected {list(sample.columns)}"
                )
            # Check row count
            if len(sub) != len(sample):
                errors.append(
                    f"Row count mismatch: got {len(sub)}, expected {len(sample)}"
                )
        else:
            warnings.append("sample_submission.csv not found — skipping column/row count checks")

        # Check for NaN in prediction columns (all columns except first ID col)
        pred_cols = sub.columns[1:] if len(sub.columns) > 1 else sub.columns
        nan_counts = sub[pred_cols].isnull().sum()
        nan_cols = nan_counts[nan_counts > 0]
        if not nan_cols.empty:
            errors.append(f"NaN values in prediction columns: {nan_cols.to_dict()}")

        if errors:
            return "VALIDATION FAILED:\n" + "\n".join(f"  - {e}" for e in errors)

        msg = f"VALIDATION PASSED: {len(sub)} rows, columns={list(sub.columns)}"
        if warnings:
            msg += "\nWarnings:\n" + "\n".join(f"  - {w}" for w in warnings)
        return msg
    except Exception as e:
        return f"Validation error: {e}"
