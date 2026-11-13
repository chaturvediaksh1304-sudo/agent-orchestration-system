"""Benchmark driver.

    python -m bench.run --set smoke --run-tag smoke
    python -m bench.run --set pilot --run-tag pilot
    caffeinate -dims python -m bench.run --set all --run-tag full

Resumable: each result is written to <results>/<arm>/<task_id>.json the moment
it exists, and a task with a result file is skipped on restart.

One task runs at a time. That keeps the token and latency numbers attributable,
and gives both arms the whole model server to themselves.
"""

import argparse
import datetime
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from bench import config
from bench.classify import classify
from bench.dataset import Task, build_prompt, load_tasks, select, unsupported_reason
from bench.proxy import UsageProxy
from bench.runner import run_child
from bench.scorer import extract_final_answer

ARMS = ("baseline", "full")


def child_env() -> Dict[str, str]:
    """The environment both arms and the worker run in."""
    env = dict(os.environ)
    env.update(
        OPENAI_BASE_URL=f"{config.PROXY_URL}/v1",
        OPENAI_API_KEY="ollama",
        CELERY_BROKER_URL=config.BROKER_URL,
        # Nothing about GAIA leaves this machine except the search queries.
        LANGCHAIN_TRACING_V2="false",
        LANGSMITH_TRACING="false",
        ANONYMIZED_TELEMETRY="False",
        # macOS aborts forked workers that touch network libraries without this.
        OBJC_DISABLE_INITIALIZE_FORK_SAFETY="YES",
        PYTHONWARNINGS="ignore",
    )
    env.pop("OPENAI_API_BASE", None)
    return env


class Worker:
    """The Celery worker pool, started exactly as the README documents it."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.process: Optional[subprocess.Popen] = None

    def start(self) -> None:
        from agent_orchestration.queue import app

        log = open(self.log_path, "ab")
        self.process = subprocess.Popen(
            [sys.executable, "-m", "celery", "-A", "agent_orchestration.queue", "worker",
             "--concurrency", str(config.WORKER_CONCURRENCY), "--loglevel", "warning",
             "--without-gossip", "--without-mingle", "--without-heartbeat"],
            stdout=log, stderr=subprocess.STDOUT, env=child_env(), cwd=config.REPO,
            start_new_session=True,
        )
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"Celery worker exited on startup; see {self.log_path}")
            if app.control.ping(timeout=1):
                return
            time.sleep(0.5)
        raise RuntimeError("Celery worker did not answer a ping within 90s")

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            # A cold kill on purpose: a warm shutdown would wait for the very
            # tasks we are trying to get rid of.
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.process.wait()
        self.process = None

    def restart(self) -> None:
        self.stop()
        flush_broker()
        self.start()


def ensure_model_idle() -> None:
    """Refuse to start the next task while the model server is still busy.

    A one-token request straight to Ollama answers in under a second when
    nothing else is generating. If it does not, the leftover load would be
    billed to the next task, so the model is unloaded and the check repeated;
    if that fails too the run stops rather than record contaminated numbers.
    """
    import urllib.request

    def responsive(timeout_s: float) -> bool:
        request = urllib.request.Request(
            f"{config.OLLAMA_URL}/v1/chat/completions",
            data=json.dumps({"model": config.MODEL, "max_tokens": 1, "temperature": 0,
                             "messages": [{"role": "user", "content": "ok"}]}).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            urllib.request.urlopen(request, timeout=timeout_s).read()
            return True
        except Exception:
            return False

    if responsive(60):
        return
    print("model server still busy after abort; unloading the model", flush=True)
    subprocess.run(["ollama", "stop", config.MODEL], capture_output=True, timeout=120)
    if not responsive(180):
        raise SystemExit("Model server did not become idle; stopping instead of measuring through it.")


def flush_broker() -> None:
    """Empty the queue, including messages a killed worker had reserved."""
    import redis

    redis.Redis.from_url(config.BROKER_URL).flushdb()


def preflight(arms: List[str]) -> None:
    import urllib.request

    with urllib.request.urlopen(f"{config.OLLAMA_URL}/v1/models", timeout=10) as response:
        models = [m["id"].replace(":latest", "") for m in json.load(response)["data"]]
    for needed in (config.MODEL, config.EMBED_MODEL):
        if needed not in models:
            raise SystemExit(f"Ollama does not have {needed!r}; run `ollama pull`/`ollama create`.")
    if "full" in arms:
        import psycopg
        import redis

        redis.Redis.from_url(config.BROKER_URL).ping()
        psycopg.connect(config.DATABASE_URL, connect_timeout=5).close()


def bench_dsn(run_tag: str) -> str:
    """A Postgres schema per run, so each run starts with empty task history."""
    import psycopg

    schema = "bench_" + "".join(c if c.isalnum() else "_" for c in run_tag)
    with psycopg.connect(config.DATABASE_URL, autocommit=True) as conn:
        conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
    return f"{config.DATABASE_URL}?options=-csearch_path%3D{schema}"


def run_one(task: Task, arm: str, state: Path, proxy: UsageProxy,
            worker: Optional[Worker], dsn: Optional[str]) -> Dict:
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "llm_calls": 0}
    reason = unsupported_reason(task)
    if reason:
        record: Dict = {"unsupported_reason": reason, "latency_s": 0.0}
    else:
        workspace = state / "workspaces" / arm / task.task_id
        shutil.rmtree(workspace, ignore_errors=True)
        workspace.mkdir(parents=True)
        if task.attachment:
            # Copied because the file tool refuses anything outside its workspace.
            shutil.copy(task.attachment, workspace / task.file_name)

        scratch = state / "scratch" / arm
        scratch.mkdir(parents=True, exist_ok=True)
        spec_path, out_path = scratch / f"{task.task_id}.spec.json", scratch / f"{task.task_id}.out.json"
        spec_path.write_text(json.dumps({
            "task_id": task.task_id, "arm": arm, "prompt": build_prompt(task),
            "workspace": str(workspace), "model": config.MODEL,
            "dsn": dsn, "chroma_dir": str(state / "chroma"),
        }))
        log_path = state / "logs" / arm / f"{task.task_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        proxy.reset()
        record = run_child(
            [sys.executable, "-m", "bench.run_task", str(spec_path), str(out_path)],
            out_path, config.TASK_TIMEOUT_S, env=child_env(), log_path=log_path, cwd=config.REPO,
        )
        usage = proxy.snapshot()

        if record.get("timed_out") or not record.get("finished"):
            # Whatever the dead run left behind must not leak into the next task.
            proxy.abort()
            if arm == "full" and worker:
                worker.restart()
            ensure_model_idle()

    label = classify(record, task.ground_truth)
    final_text = record.get("final_text") or ""
    return {
        "task_id": task.task_id,
        "level": task.level,
        "arm": arm,
        "outcome": label["outcome"],
        "sub_cause": label["sub_cause"],
        "final_answer": label["final_answer"],
        "ground_truth": task.ground_truth,
        "latency_s": record.get("latency_s", 0.0),
        "tokens_in": usage["prompt_tokens"],
        "tokens_out": usage["completion_tokens"],
        "llm_calls": usage["llm_calls"],
        "retries_triggered": record.get("retries_triggered"),
        "subagents_spawned": record.get("subagents_spawned"),
        "error_type": record.get("error_type"),
        "error_message": record.get("error_message"),
        "timed_out": bool(record.get("timed_out")),
        "status": record.get("status"),
        "answer_had_marker": extract_final_answer(final_text)[1] if final_text else None,
        "final_text": final_text[:4000],
        "subtasks": record.get("subtasks"),
        "has_attachment": bool(task.file_name),
        "model": config.MODEL,
        "finished_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", default="smoke", help="smoke | pilot | all | balanced:N")
    parser.add_argument("--run-tag", required=True, help="Names this run's state (memory, logs).")
    parser.add_argument("--results-dir", default=None, help="Default: results for --run-tag full, else results-<tag>.")
    parser.add_argument("--arms", default="baseline,full")
    args = parser.parse_args(argv)

    arms = [a for a in args.arms.split(",") if a in ARMS]
    results_dir = Path(args.results_dir or ("results" if args.run_tag == "full" else f"results-{args.run_tag}"))
    state = config.STATE_DIR / args.run_tag
    (state / "logs").mkdir(parents=True, exist_ok=True)

    tasks = select(load_tasks(), args.set)
    preflight(arms)
    dsn = bench_dsn(args.run_tag) if "full" in arms else None

    proxy = UsageProxy()
    proxy.start()
    worker = Worker(state / "logs" / "worker.log") if "full" in arms else None
    if worker:
        flush_broker()
        worker.start()

    print(f"{len(tasks)} tasks x {len(arms)} arms -> {results_dir}/  (model {config.MODEL})", flush=True)
    try:
        for position, task in enumerate(tasks):
            # Alternate which arm goes first, so neither always sees a warm cache.
            order = arms if position % 2 == 0 else list(reversed(arms))
            for arm in order:
                path = results_dir / arm / f"{task.task_id}.json"
                if path.exists():
                    continue
                result = run_one(task, arm, state, proxy, worker, dsn)
                result["position"] = position
                path.parent.mkdir(parents=True, exist_ok=True)
                temp = path.with_suffix(".tmp")
                temp.write_text(json.dumps(result, indent=2, ensure_ascii=False))
                os.replace(temp, path)
                print(
                    f"[{position + 1:3d}/{len(tasks)}] L{task.level} {arm:8s} {task.task_id[:8]} "
                    f"{result['outcome']:22s} {result['sub_cause'] or '':24s} "
                    f"{result['latency_s']:7.1f}s  in={result['tokens_in']} out={result['tokens_out']}",
                    flush=True,
                )
    finally:
        if worker:
            worker.stop()
        proxy.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
