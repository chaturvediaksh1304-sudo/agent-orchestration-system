"""Phase 3 criterion 2: multiple subagents execute concurrently.

Needs a real Redis broker and real Celery worker processes, so it is excluded
from the default run:

    pytest -m integration

Concurrency is measured with a sleeping probe task rather than a real subagent,
so this proves the queue without needing an API key. The probe reports its pid,
so the test distinguishes real parallelism from a fast sequential run.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import redis

from agent_orchestration.queue import BROKER_URL, app

TASKS = 4
SLEEP_SECONDS = 2.0
PROBE_QUEUE = "test_probe"
TESTS_DIR = Path(__file__).parent


@pytest.fixture(scope="module")
def broker():
    try:
        redis.Redis.from_url(BROKER_URL).ping()
    except redis.exceptions.ConnectionError:
        pytest.skip(f"No Redis at {BROKER_URL}; start one to run integration tests.")
    return BROKER_URL


@pytest.fixture(scope="module")
def worker(broker):
    """A single worker with a pool of TASKS processes."""
    process = subprocess.Popen(
        [
            sys.executable, "-m", "celery",
            "-A", "worker_probe", "worker",
            # A dedicated queue: the compose worker consumes the default one and
            # would steal probe tasks it has no registration for.
            "-Q", PROBE_QUEUE,
            "--concurrency", str(TASKS),
            "--loglevel", "error",
            "--without-gossip", "--without-mingle", "--without-heartbeat",
        ],
        cwd=TESTS_DIR,
        # Inherit the environment: a stripped one drops CELERY_BROKER_URL, so the
        # worker would silently fall back to the default broker while the test
        # enqueues to the configured one.
        env={**os.environ, "PYTHONPATH": str(TESTS_DIR)},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    _wait_until_ready(process)
    yield process
    process.terminate()
    process.wait(timeout=30)


def _wait_until_ready(process, timeout=60):
    """Poll the control plane until a worker answers, so timing isn't skewed."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            pytest.fail(f"Worker died during startup:\n{process.stdout.read().decode()}")
        if app.control.ping(timeout=1):
            return
        time.sleep(0.5)
    process.terminate()
    pytest.fail(f"Worker did not become ready within {timeout}s")


@pytest.mark.integration
def test_tasks_run_concurrently_not_sequentially(worker):
    from worker_probe import sleep_probe

    started = time.time()
    results = [sleep_probe.apply_async(args=[SLEEP_SECONDS], queue=PROBE_QUEUE) for _ in range(TASKS)]
    finished = [r.get(timeout=60) for r in results]
    elapsed = time.time() - started

    sequential = TASKS * SLEEP_SECONDS
    assert len(finished) == TASKS
    assert elapsed < sequential * 0.6, (
        f"{TASKS} x {SLEEP_SECONDS}s took {elapsed:.1f}s; "
        f"sequential would be {sequential:.1f}s, so this did not run concurrently."
    )


@pytest.mark.integration
def test_tasks_are_spread_across_worker_processes(worker):
    """Distinct pids prove real parallelism rather than a lucky fast run."""
    from worker_probe import sleep_probe

    results = [sleep_probe.apply_async(args=[SLEEP_SECONDS], queue=PROBE_QUEUE) for _ in range(TASKS)]
    finished = [r.get(timeout=60) for r in results]

    assert len({r["pid"] for r in finished}) > 1


@pytest.mark.integration
def test_execution_windows_actually_overlap(worker):
    """The strongest check: some task started before another had finished."""
    from worker_probe import sleep_probe

    results = [sleep_probe.apply_async(args=[SLEEP_SECONDS], queue=PROBE_QUEUE) for _ in range(TASKS)]
    finished = sorted((r.get(timeout=60) for r in results), key=lambda r: r["started"])

    assert any(
        later["started"] < earlier["finished"]
        for earlier, later in zip(finished, finished[1:])
    )


@pytest.mark.integration
def test_self_repair_runs_inside_the_worker(worker):
    """Phase 4 through the real queue: a worker repairs its own failure."""
    from worker_probe import repair_probe

    outcome = repair_probe.apply_async(kwargs={"fail_times": 1, "max_attempts": 3}, queue=PROBE_QUEUE).get(timeout=60)

    assert outcome["ok"]
    assert outcome["output"] == "recovered in the worker"
    assert outcome["attempts"] == 2
    assert outcome["failures"][0]["diagnosis"] == "diagnosed cause 1"


@pytest.mark.integration
def test_exhausted_repair_crosses_the_worker_boundary_intact(worker):
    """ok=False and the full failure history must survive JSON serialisation."""
    from worker_probe import repair_probe

    outcome = repair_probe.apply_async(kwargs={"fail_times": 5, "max_attempts": 3}, queue=PROBE_QUEUE).get(timeout=60)

    assert outcome["ok"] is False
    assert outcome["attempts"] == 3
    assert len(outcome["failures"]) == 3
    assert outcome["failures"][-1]["diagnosis"] is None


@pytest.mark.integration
def test_results_come_back_in_submission_order(worker):
    """Criterion 3 against the real broker: order is by submission, not completion."""
    from worker_probe import sleep_probe

    # Descending sleeps: completion order is the reverse of submission order.
    delays = [1.5, 1.0, 0.5, 0.1]
    results = [sleep_probe.apply_async(args=[d], queue=PROBE_QUEUE) for d in delays]
    finished = [r.get(timeout=60) for r in results]

    durations = [r["finished"] - r["started"] for r in finished]
    assert durations == sorted(durations, reverse=True)
