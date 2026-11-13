"""Every knob of the experiment in one place. Both arms read the same values."""

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# GAIA stays where it was downloaded; it is never copied into the repo.
GAIA_DIR = Path(os.environ.get("GAIA_DIR", Path.home() / "gaia-test" / "2023" / "validation"))
# Workspaces, logs, Chroma files. Outside the repo (and outside iCloud).
STATE_DIR = Path(os.environ.get("BENCH_STATE_DIR", Path.home() / "gaia-test" / "bench-state"))

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
# qwen2.5:7b weights (Q4_K_M) with num_ctx raised from Ollama's 4096 default.
MODEL = os.environ.get("BENCH_MODEL", "qwen2.5-7b-ctx16k")
EMBED_MODEL = "nomic-embed-text"

# The harness counts tokens by sitting between the system and Ollama.
PROXY_PORT = int(os.environ.get("BENCH_PROXY_PORT", "11500"))
PROXY_URL = f"http://localhost:{PROXY_PORT}"

TASK_TIMEOUT_S = 20 * 60
MAX_ATTEMPTS = 3  # the system's own default repair budget
WORKER_CONCURRENCY = 4  # as documented in the README and docker-compose.yml
SEED = 20261109

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://orchestration:orchestration@localhost:5432/orchestration"
)
BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
