# Python 3.9 to match the pinned dependency set (see Memory.md for why).
FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first so edits to source don't invalidate the install layer.
COPY pyproject.toml ./
COPY agent_orchestration/__init__.py ./agent_orchestration/
RUN pip install --no-cache-dir -e .

COPY agent_orchestration/ ./agent_orchestration/

# Subagent file tools are confined to this directory.
RUN mkdir -p /app/workspace /app/.orchestration

ENTRYPOINT ["python", "-m", "agent_orchestration.cli"]
CMD ["--help"]
