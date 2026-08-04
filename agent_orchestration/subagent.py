"""The subagent factory.

This is the only place in the codebase where a subagent is constructed, and it
knows nothing about any particular role — everything that distinguishes one
subagent from another arrives in the AgentSpec at runtime. That is what
"dynamically generated, not hardcoded" means here.
"""

from typing import Dict

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent

from agent_orchestration.state import AgentSpec


def build_subagent(
    spec: AgentSpec, llm: BaseChatModel, registry: Dict[str, BaseTool]
) -> CompiledStateGraph:
    """Turn a runtime-authored AgentSpec into a runnable agent."""
    # The spec is model-authored, so it can name tools that don't exist.
    # Filtering keeps a hallucinated name from killing an otherwise valid run.
    tools = [registry[name] for name in spec.tools if name in registry]
    return create_react_agent(llm, tools=tools, prompt=spec.system_prompt)
