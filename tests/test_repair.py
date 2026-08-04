"""Phase 4: a failing subagent diagnoses itself, retries, and gives up cleanly.

All three done-criteria live here:
  1. a deliberately failing subagent triggers a diagnosis step
  2. at least one automated retry/fix is attempted, based on that diagnosis
  3. repeated failure after those attempts is correctly detected
"""

from conftest import diagnosis_message
from langchain_core.messages import AIMessage

from agent_orchestration.repair import execute_with_repair
from agent_orchestration.state import AgentSpec
from agent_orchestration.tools import build_tools

SPEC = AgentSpec(role="writer", system_prompt="You write.", tools=["write_file"])


def test_a_run_that_succeeds_first_time_needs_no_repair(stub_model, tmp_path):
    llm = stub_model([AIMessage(content="done first time")])

    outcome = execute_with_repair(SPEC, "do it", llm, build_tools(tmp_path))

    assert outcome.ok
    assert outcome.output == "done first time"
    assert outcome.attempts == 1
    assert outcome.failures == []


def test_failure_triggers_a_diagnosis_step(stub_model, tmp_path):
    """Criterion 1."""
    llm = stub_model(
        [
            RuntimeError("tool exploded"),
            diagnosis_message("the agent lacked the right tool"),
            AIMessage(content="done on retry"),
        ]
    )

    outcome = execute_with_repair(SPEC, "do it", llm, build_tools(tmp_path))

    assert outcome.failures[0].diagnosis == "the agent lacked the right tool"
    assert "tool exploded" in outcome.failures[0].error


def test_retry_after_diagnosis_succeeds(stub_model, tmp_path):
    """Criterion 2."""
    llm = stub_model(
        [
            RuntimeError("first attempt failed"),
            diagnosis_message("bad prompt"),
            AIMessage(content="recovered"),
        ]
    )

    outcome = execute_with_repair(SPEC, "do it", llm, build_tools(tmp_path))

    assert outcome.ok
    assert outcome.output == "recovered"
    assert outcome.attempts == 2


def test_the_retry_uses_the_diagnosed_prompt_not_the_original(stub_model, tmp_path):
    """A retry that ignored the diagnosis would just repeat the same failure."""
    llm = stub_model(
        [
            RuntimeError("boom"),
            diagnosis_message("prompt was too vague", prompt="Be extremely specific."),
            AIMessage(content="ok"),
        ]
    )

    execute_with_repair(SPEC, "do it", llm, build_tools(tmp_path))

    assert "Be extremely specific." in str(llm.seen_messages[-1])


def test_the_retry_uses_the_diagnosed_tool_set(stub_model, tmp_path):
    llm = stub_model(
        [
            RuntimeError("boom"),
            diagnosis_message("needed to read, not write", tools=["read_file", "list_files"]),
            AIMessage(content="ok"),
        ]
    )

    execute_with_repair(SPEC, "do it", llm, build_tools(tmp_path))

    assert llm.bound_tool_names == ["read_file", "list_files"]


def test_repeated_failure_is_detected_and_reported(stub_model, tmp_path):
    """Criterion 3: give up cleanly rather than looping or raising."""
    llm = stub_model(
        [
            RuntimeError("fail 1"),
            diagnosis_message("cause 1"),
            RuntimeError("fail 2"),
            diagnosis_message("cause 2"),
            RuntimeError("fail 3"),
        ]
    )

    outcome = execute_with_repair(SPEC, "do it", llm, build_tools(tmp_path), max_attempts=3)

    assert not outcome.ok
    assert outcome.attempts == 3
    assert [f.error.split(":")[-1].strip() for f in outcome.failures] == [
        "fail 1",
        "fail 2",
        "fail 3",
    ]


def test_the_final_failure_is_not_diagnosed(stub_model, tmp_path):
    """Diagnosing a failure you will never retry is a wasted model call."""
    llm = stub_model(
        [RuntimeError("fail 1"), diagnosis_message("cause 1"), RuntimeError("fail 2")]
    )

    outcome = execute_with_repair(SPEC, "do it", llm, build_tools(tmp_path), max_attempts=2)

    assert outcome.failures[-1].diagnosis is None


def test_max_attempts_of_one_means_no_repair_at_all(stub_model, tmp_path):
    llm = stub_model([RuntimeError("the only attempt")])

    outcome = execute_with_repair(SPEC, "do it", llm, build_tools(tmp_path), max_attempts=1)

    assert not outcome.ok
    assert outcome.attempts == 1
    assert outcome.failures[0].diagnosis is None


def test_a_failing_diagnosis_does_not_crash_the_run(stub_model, tmp_path):
    """The repair machinery must not become a new source of failure."""
    llm = stub_model([RuntimeError("boom"), RuntimeError("the diagnosis itself failed")])

    outcome = execute_with_repair(SPEC, "do it", llm, build_tools(tmp_path), max_attempts=3)

    assert not outcome.ok
    assert "the diagnosis itself failed" in outcome.failures[-1].error


def test_outcome_survives_json_round_trip(stub_model, tmp_path):
    """Outcomes cross the Celery boundary, where only JSON is accepted."""
    import json

    from agent_orchestration.repair import SubtaskOutcome

    llm = stub_model([RuntimeError("boom"), diagnosis_message("cause"), AIMessage(content="ok")])
    outcome = execute_with_repair(SPEC, "do it", llm, build_tools(tmp_path))

    restored = SubtaskOutcome(**json.loads(json.dumps(outcome.model_dump())))

    assert restored.ok
    assert restored.failures[0].diagnosis == "cause"


def test_a_real_tool_failure_is_repaired(stub_model, tmp_path):
    """End to end through the actual tool guard, not a synthetic exception."""
    llm = stub_model(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_file",
                        "args": {"path": "../escape.txt", "content": "x"},
                        "id": "c1",
                    }
                ],
            ),
            diagnosis_message("tried to write outside the workspace"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_file",
                        "args": {"path": "inside.txt", "content": "x"},
                        "id": "c2",
                    }
                ],
            ),
            AIMessage(content="wrote it properly"),
        ]
    )

    outcome = execute_with_repair(SPEC, "write a file", llm, build_tools(tmp_path))

    assert outcome.ok
    assert (tmp_path / "inside.txt").exists()
