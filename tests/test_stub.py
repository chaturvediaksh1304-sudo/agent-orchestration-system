"""Guards the test harness itself.

Every other test drives the graph through StubChatModel, so if the stub stops
honouring the BaseChatModel contract, those tests fail for reasons that have
nothing to do with the code under test. These three cases pin the contract.
"""

import pytest
from langchain_core.messages import AIMessage
from pydantic import BaseModel


class _Answer(BaseModel):
    value: str


def test_returns_scripted_text_in_order(stub_model):
    llm = stub_model(["first", "second"])

    assert llm.invoke("anything").content == "first"
    assert llm.invoke("anything").content == "second"


def test_bind_tools_records_what_it_was_given(stub_model):
    llm = stub_model(["ok"])

    llm.bind_tools([{"name": "read_file", "description": "d", "parameters": {}}])

    assert llm.bound_tool_names == ["read_file"]


def test_scripted_exception_is_raised(stub_model):
    """Drives the self-repair loop: a deliberately failing model call."""
    llm = stub_model([RuntimeError("the model fell over"), "recovered"])

    with pytest.raises(RuntimeError, match="the model fell over"):
        llm.invoke("anything")

    assert llm.invoke("anything").content == "recovered"


def test_drives_with_structured_output(stub_model):
    """The path the supervisor's decompose step depends on."""
    llm = stub_model(
        [AIMessage(content="", tool_calls=[{"name": "_Answer", "args": {"value": "hi"}, "id": "1"}])]
    )

    result = llm.with_structured_output(_Answer).invoke("anything")

    assert isinstance(result, _Answer)
    assert result.value == "hi"
