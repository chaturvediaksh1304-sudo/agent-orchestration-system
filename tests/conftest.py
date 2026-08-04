import hashlib
import math
from typing import Any, Dict, List, Optional, Sequence, Union

import pytest
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import PrivateAttr


class StubChatModel(BaseChatModel):
    """Replays scripted responses instead of calling a model.

    ``bind_tools`` is overridden deliberately: ``BaseChatModel.with_structured_output``
    raises NotImplementedError unless a subclass provides its own, then routes the
    schema through ``bind_tools`` + PydanticToolsParser. Overriding it is what makes
    the supervisor's structured-output path testable without an API call.
    """

    # Any, not Union[str, AIMessage, BaseException]: pydantic's smart union tries
    # to coerce an exception into an AIMessage and blows up inside its validator.
    # _generate dispatches on isinstance at runtime instead.
    responses: List[Any]
    bound_tool_names: List[str] = []
    seen_messages: List[List[BaseMessage]] = []
    _calls: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return "stub"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.seen_messages.append(messages)
        if self._calls >= len(self.responses):
            raise AssertionError(
                f"StubChatModel ran out of scripted responses after {self._calls} "
                f"call(s); the code under test called the model more times than the "
                f"test expected."
            )
        response = self.responses[self._calls]
        self._calls += 1
        if isinstance(response, BaseException):
            # Lets a test script a deliberately failing subagent, which is what
            # the self-repair loop needs to be driven against.
            raise response
        message = AIMessage(content=response) if isinstance(response, str) else response
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(
        self, tools: Sequence[Union[Dict[str, Any], type, Any]], **kwargs: Any
    ) -> Runnable[Any, BaseMessage]:
        self.bound_tool_names = [
            convert_to_openai_tool(tool)["function"]["name"] for tool in tools
        ]
        return self.bind(tools=list(tools), **kwargs)


@pytest.fixture
def stub_model():
    """Returns a factory so each test scripts its own responses."""
    return lambda responses: StubChatModel(responses=responses)


def decomposition_message(*pairs) -> AIMessage:
    """Build an AIMessage carrying a Decomposition, as with_structured_output expects.

    Each pair is (description, role, tools).
    """
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "Decomposition",
                "args": {
                    "subtasks": [
                        {
                            "description": description,
                            "spec": {
                                "role": role,
                                "system_prompt": f"You are the {role}.",
                                "tools": tools,
                            },
                        }
                        for description, role, tools in pairs
                    ]
                },
                "id": "call_decompose",
            }
        ],
    )


class StubEmbeddingFunction(EmbeddingFunction):
    """Deterministic hashing vectoriser, so tests never download or call a model.

    Chroma 1.x requires name/get_config/build_from_config alongside __call__ for
    an embedding function to be usable and persistable.
    """

    DIMENSIONS = 64

    def __init__(self) -> None:
        pass

    def __call__(self, input: Documents) -> Embeddings:
        return [self._vector(document) for document in input]

    @classmethod
    def _vector(cls, text: str) -> List[float]:
        vector = [0.0] * cls.DIMENSIONS
        for token in text.lower().split():
            # md5 rather than hash(): Python salts str hashing per process, which
            # would make persisted embeddings unreadable after a restart.
            digest = hashlib.md5(token.encode()).hexdigest()
            vector[int(digest, 16) % cls.DIMENSIONS] += 1.0
        magnitude = math.sqrt(sum(v * v for v in vector))
        return [v / magnitude for v in vector] if magnitude else vector

    @staticmethod
    def name() -> str:
        return "stub"

    def get_config(self) -> Dict[str, Any]:
        return {}

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "StubEmbeddingFunction":
        return StubEmbeddingFunction()


@pytest.fixture
def stub_embeddings() -> StubEmbeddingFunction:
    return StubEmbeddingFunction()


@pytest.fixture
def store():
    """A Store on a scratch Postgres schema, dropped afterwards.

    Since the Phase 6 swap the store needs a real database, so anything using
    this fixture is an integration test. Each test gets its own schema so they
    stay independent and can run in any order.
    """
    import uuid

    import psycopg

    from agent_orchestration.store import DEFAULT_DSN, Store

    try:
        psycopg.connect(DEFAULT_DSN, connect_timeout=3).close()
    except psycopg.OperationalError:
        pytest.skip(f"No Postgres at {DEFAULT_DSN}; run `docker compose up -d postgres`.")

    schema = f"test_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(DEFAULT_DSN, autocommit=True) as conn:
        conn.execute(f'CREATE SCHEMA "{schema}"')
    try:
        yield Store(f"{DEFAULT_DSN}?options=-csearch_path%3D{schema}")
    finally:
        with psycopg.connect(DEFAULT_DSN, autocommit=True) as conn:
            conn.execute(f'DROP SCHEMA "{schema}" CASCADE')


def diagnosis_message(cause, prompt="Try again, more carefully.", tools=None) -> AIMessage:
    """An AIMessage carrying a Diagnosis, as with_structured_output expects."""
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "Diagnosis",
                "args": {
                    "cause": cause,
                    "revised_system_prompt": prompt,
                    "revised_tools": ["write_file"] if tools is None else tools,
                },
                "id": "call_diagnose",
            }
        ],
    )


@pytest.fixture
def two_subtasks() -> AIMessage:
    """A writer-then-summariser decomposition, the shared multi-step fixture."""
    return decomposition_message(
        ("Write the facts to a file", "writer", ["write_file"]),
        ("Read the file back and summarise it", "summariser", ["read_file"]),
    )
