"""Done-criterion 2: subagents are dynamically generated, not hardcoded.

The claim these tests defend is structural: one generic factory turns any
runtime-authored AgentSpec into an agent, so nothing about a subagent's role
or tool access is fixed in the source.
"""

from langchain_core.messages import AIMessage

from agent_orchestration.state import AgentSpec
from agent_orchestration.subagent import build_subagent
from agent_orchestration.tools import build_tools


def test_spec_tool_subset_is_what_the_agent_gets(stub_model, tmp_path):
    spec = AgentSpec(
        role="writer", system_prompt="You write files.", tools=["write_file", "list_files"]
    )
    llm = stub_model(["ok"])

    build_subagent(spec, llm, build_tools(tmp_path))

    assert llm.bound_tool_names == ["write_file", "list_files"]


def test_two_different_specs_yield_two_different_agents(stub_model, tmp_path):
    """Same factory, same call site — the difference comes only from the spec."""
    registry = build_tools(tmp_path)
    reader = AgentSpec(role="reader", system_prompt="You read.", tools=["read_file"])
    writer = AgentSpec(role="writer", system_prompt="You write.", tools=["write_file"])

    reader_llm, writer_llm = stub_model(["a"]), stub_model(["b"])
    build_subagent(reader, reader_llm, registry)
    build_subagent(writer, writer_llm, registry)

    assert reader_llm.bound_tool_names == ["read_file"]
    assert writer_llm.bound_tool_names == ["write_file"]


def test_hallucinated_tool_names_are_filtered_not_fatal(stub_model, tmp_path):
    """A model-authored spec can name tools that don't exist; that must not crash."""
    spec = AgentSpec(
        role="researcher",
        system_prompt="You research.",
        tools=["read_file", "search_the_web", "send_email"],
    )
    llm = stub_model(["ok"])

    build_subagent(spec, llm, build_tools(tmp_path))

    assert llm.bound_tool_names == ["read_file"]


def test_spec_naming_no_valid_tools_still_builds(stub_model, tmp_path):
    spec = AgentSpec(role="thinker", system_prompt="You think.", tools=["nonexistent"])

    agent = build_subagent(spec, stub_model(["done"]), build_tools(tmp_path))

    assert agent is not None


def test_agent_actually_executes_a_tool(stub_model, tmp_path):
    """Done-criterion 2's second half: the subagent can call at least one tool."""
    spec = AgentSpec(role="writer", system_prompt="You write.", tools=["write_file"])
    llm = stub_model(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_file",
                        "args": {"path": "out.txt", "content": "written by tool"},
                        "id": "call_1",
                    }
                ],
            ),
            AIMessage(content="Done."),
        ]
    )
    agent = build_subagent(spec, llm, build_tools(tmp_path))

    agent.invoke({"messages": [("user", "write the file")]})

    assert (tmp_path / "out.txt").read_text() == "written by tool"


def test_system_prompt_from_spec_reaches_the_model(stub_model, tmp_path):
    spec = AgentSpec(
        role="pirate", system_prompt="Answer only in pirate speak.", tools=[]
    )
    llm = stub_model([AIMessage(content="Arrr.")])
    agent = build_subagent(spec, llm, build_tools(tmp_path))

    agent.invoke({"messages": [("user", "hello")]})

    assert "Answer only in pirate speak." in str(llm.seen_messages)
