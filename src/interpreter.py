"""
Persistent Python interpreter — runs code in a long-lived subprocess.
Variables, imports, and trained objects are preserved between calls.
"""
import subprocess
import sys
import textwrap
import threading
import uuid


TIMEOUT = 120  # seconds per code block


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
            sentinel = f"__DONE_{uuid.uuid4().hex}__"
            wrapped = textwrap.dedent(f"""
try:
    exec(compile({repr(code)}, '<cell>', 'exec'), __globals__)
except Exception as _e:
    import traceback
    print(traceback.format_exc(), end='')
print({repr(sentinel)})
""")
            self._proc.stdin.write(wrapped)
            self._proc.stdin.flush()

            lines = []
            while True:
                line = self._read_line_timeout(TIMEOUT)
                if line is None:
                    return "\n".join(lines) + "\n[TIMEOUT: code took too long]"
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


_REPL_BOOTSTRAP = """
import sys, os
__globals = {}
while True:
    code = sys.stdin.readline()
    if not code:
        break
    exec(compile(code, '<input>', 'exec'), __globals)
"""
