"""Run one task attempt in its own process, under a hard wall-clock limit.

A separate process is what makes the timeout real: a hung model call or a
runaway agent loop is killed from outside, and a crash cannot take the
benchmark driver down with it.
"""

import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional


def run_child(
    cmd: List[str],
    out_path: Path,
    timeout_s: float,
    env: Optional[Dict[str, str]] = None,
    log_path: Optional[Path] = None,
    cwd: Optional[Path] = None,
) -> Dict:
    """Run ``cmd`` and return what it reported, plus what the harness observed.

    The child writes its findings to ``out_path`` as JSON. Whatever happens —
    clean exit, exception, kill — the returned record says so explicitly.
    """
    out_path = Path(out_path)
    if out_path.exists():
        out_path.unlink()
    log = open(log_path, "ab") if log_path else subprocess.DEVNULL

    started = time.monotonic()
    # Its own session, so the kill reaches any grandchildren too.
    process = subprocess.Popen(
        cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=cwd, start_new_session=True
    )
    timed_out = False
    try:
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
    latency = time.monotonic() - started
    if log_path:
        log.close()

    record: Dict = {}
    if out_path.exists():
        try:
            record = json.loads(out_path.read_text())
        except ValueError:
            record = {}

    record["timed_out"] = timed_out
    record["latency_s"] = round(latency, 3)
    record["exit_code"] = process.returncode
    if not timed_out and not record.get("finished") and not record.get("error_type"):
        # Died without reporting anything: segfault, OOM kill, import failure.
        record["error_type"] = "ChildProcessDied"
        record["error_message"] = f"exited with code {process.returncode} without a result"
    return record
