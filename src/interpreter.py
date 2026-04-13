"""
Persistent Python interpreter — runs code in a long-lived subprocess.
Variables, imports, and trained objects are preserved between calls.
"""
import subprocess
import sys
import threading
import uuid


TIMEOUT = 300  # seconds per code block


class PersistentInterpreter:
    def __init__(self):
        self._proc = subprocess.Popen(
            [sys.executable, "-u", "-c", _REPL_BOOTSTRAP],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self._lock = threading.Lock()

    def run(self, code: str) -> str:
        """Execute code and return combined stdout+stderr output."""
        with self._lock:
            if self._proc.poll() is not None:
                return "[ERROR: interpreter process has terminated]"

            sentinel = f"__DONE_{uuid.uuid4().hex}__"

            # Build a try/except block that catches errors and prints sentinel
            inner = (
                f"try:\n"
                f"    exec(compile({repr(code)}, '<cell>', 'exec'), globals())\n"
                f"except Exception as _e:\n"
                f"    import traceback; print(traceback.format_exc(), end='')\n"
                f"print({repr(sentinel)})\n"
            )
            # Wrap in a single-line exec() so the REPL can read it via readline()
            one_liner = f"exec(compile({repr(inner)}, '<wrapped>', 'exec'))\n"

            self._proc.stdin.write(one_liner)
            self._proc.stdin.flush()

            lines = []
            while True:
                line = self._read_line_timeout(TIMEOUT)
                if line is None:
                    return "\n".join(lines) + "\n[TIMEOUT: code took too long]"
                if line == "":
                    # EOF — subprocess died
                    return "\n".join(lines) + "\n[ERROR: interpreter process terminated]"
                if line.rstrip() == sentinel:
                    break
                lines.append(line.rstrip())
            return "\n".join(lines)

    def _read_line_timeout(self, timeout: float):
        result = [None]
        def _read():
            try:
                result[0] = self._proc.stdout.readline()
            except Exception:
                result[0] = None
        t = threading.Thread(target=_read, daemon=True)
        t.start()
        t.join(timeout)
        return result[0]

    def close(self):
        try:
            self._proc.stdin.close()
            self._proc.terminate()
        except Exception:
            pass


_REPL_BOOTSTRAP = """\
import sys, os
__globals = {}

# Pre-import common ML libraries
try:
    import numpy as np; __globals['np'] = np
    import pandas as pd; __globals['pd'] = pd
    from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
    __globals.update({'cross_val_score': cross_val_score, 'StratifiedKFold': StratifiedKFold, 'KFold': KFold})
    from sklearn.preprocessing import LabelEncoder
    __globals['LabelEncoder'] = LabelEncoder
    from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
    __globals.update({'accuracy_score': accuracy_score, 'roc_auc_score': roc_auc_score, 'mean_squared_error': mean_squared_error})
except Exception:
    pass

while True:
    line = sys.stdin.readline()
    if not line:
        break
    try:
        exec(compile(line, '<input>', 'exec'), __globals)
    except Exception:
        import traceback
        print(traceback.format_exc(), end='')
"""
