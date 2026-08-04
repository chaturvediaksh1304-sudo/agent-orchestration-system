"""Done-criterion 3 against a real model.

Excluded from the default run (see the `live` marker in pyproject.toml) because
it costs money and is nondeterministic. Run it with:

    pytest -m live

Assertions here check the shape of a successful run, not the model's wording.
"""

import os

import pytest

from agent_orchestration.llm import build_llm
from agent_orchestration.supervisor import build_supervisor
from agent_orchestration.tools import build_tools

GOAL = (
    "Write three facts about the Apollo program to facts.txt, "
    "then read that file back and summarise it."
)


@pytest.fixture
def provider():
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    pytest.skip("Set ANTHROPIC_API_KEY or OPENAI_API_KEY to run the live test.")


@pytest.mark.live
def test_full_run_against_a_real_model(provider, tmp_path):
    llm = build_llm(provider)
    tools = build_tools(tmp_path)

    result = build_supervisor(llm, tools).invoke({"goal": GOAL})

    # Decomposition happened, and into more than one step.
    assert len(result["subtasks"]) >= 2

    # Subagents were designed at runtime, each with a real prompt.
    for subtask in result["subtasks"]:
        assert subtask.spec.role
        assert subtask.spec.system_prompt

    # A tool actually ran: the file exists in the workspace with content.
    assert (tmp_path / "facts.txt").read_text().strip()

    # Every subtask produced output, and the run aggregated to a final answer.
    assert len(result["results"]) == len(result["subtasks"])
    assert result["final"].strip()
