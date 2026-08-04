"""Done-criteria 1 and 3, driven deterministically by the stub model.

The live counterpart in test_live_e2e.py proves the same path against a real
model; these tests pin the wiring so a break here is unambiguous.
"""

import pytest
from langchain_core.messages import AIMessage

from agent_orchestration.state import AgentSpec, Decomposition, Subtask
from agent_orchestration.supervisor import build_supervisor, decompose_goal
from agent_orchestration.tools import build_tools


def test_decompose_produces_multiple_subtasks_each_with_a_spec(stub_model, two_subtasks):
    subtasks = decompose_goal("write facts then summarise them", stub_model([two_subtasks]))

    assert len(subtasks) >= 2
    assert all(isinstance(s, Subtask) for s in subtasks)
    assert all(isinstance(s.spec, AgentSpec) for s in subtasks)
    assert [s.spec.role for s in subtasks] == ["writer", "summariser"]


def test_decompose_specs_carry_distinct_tool_subsets(stub_model, two_subtasks):
    """The supervisor decides tool access per subtask; it is not a fixed set."""
    subtasks = decompose_goal("anything", stub_model([two_subtasks]))

    assert [s.spec.tools for s in subtasks] == [["write_file"], ["read_file"]]


def test_end_to_end_run_produces_an_aggregated_result(stub_model, two_subtasks, tmp_path):
    """Done-criterion 3, deterministically: goal in, aggregated result out."""
    llm = stub_model(
        [
            two_subtasks,
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_file",
                        "args": {"path": "facts.txt", "content": "the moon is far"},
                        "id": "c1",
                    }
                ],
            ),
            AIMessage(content="Wrote the facts."),
            AIMessage(
                content="",
                tool_calls=[{"name": "read_file", "args": {"path": "facts.txt"}, "id": "c2"}],
            ),
            AIMessage(content="The file says the moon is far."),
            AIMessage(content="FINAL: facts written and summarised."),
        ]
    )

    result = build_supervisor(llm, build_tools(tmp_path)).invoke(
        {"goal": "write facts then summarise them"}
    )

    assert (tmp_path / "facts.txt").read_text() == "the moon is far"
    assert len(result["results"]) == 2
    assert result["final"] == "FINAL: facts written and summarised."


def test_each_subtask_result_is_captured(stub_model, two_subtasks, tmp_path):
    llm = stub_model(
        [
            two_subtasks,
            AIMessage(content="first subtask done"),
            AIMessage(content="second subtask done"),
            AIMessage(content="aggregated"),
        ]
    )

    result = build_supervisor(llm, build_tools(tmp_path)).invoke({"goal": "g"})

    assert [r.output for r in result["results"]] == [
        "first subtask done",
        "second subtask done",
    ]
    assert [r.subtask.spec.role for r in result["results"]] == ["writer", "summariser"]


def test_aggregation_sees_every_subtask_result(stub_model, two_subtasks, tmp_path):
    llm = stub_model(
        [
            two_subtasks,
            AIMessage(content="alpha-result"),
            AIMessage(content="beta-result"),
            AIMessage(content="aggregated"),
        ]
    )

    build_supervisor(llm, build_tools(tmp_path)).invoke({"goal": "g"})

    aggregation_prompt = str(llm.seen_messages[-1])
    assert "alpha-result" in aggregation_prompt
    assert "beta-result" in aggregation_prompt


def test_decomposition_with_no_subtasks_is_rejected(stub_model, two_subtasks):
    empty = AIMessage(
        content="",
        tool_calls=[{"name": "Decomposition", "args": {"subtasks": []}, "id": "c"}],
    )

    with pytest.raises(ValueError, match="no subtasks"):
        decompose_goal("an impossible goal", stub_model([empty]))
