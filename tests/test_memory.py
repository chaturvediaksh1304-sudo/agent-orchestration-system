"""Phase 2 done-criterion 2: conversation memory writes to and reads from ChromaDB.

Tests inject a deterministic stub embedding so the suite stays offline, free and
fast; runtime uses Chroma's default local model.
"""

import pytest

from agent_orchestration.memory import ConversationMemory


@pytest.fixture
def memory(tmp_path, stub_embeddings):
    return ConversationMemory(tmp_path / "chroma", embedding_function=stub_embeddings)


def test_remembered_run_can_be_recalled(memory):
    memory.remember(run_id=1, goal="research the apollo program", summary="Apollo 11 landed in 1969.")

    recalled = memory.recall("apollo program", limit=1)

    assert len(recalled) == 1
    assert "Apollo 11 landed in 1969." in recalled[0]


def test_recall_returns_the_closest_match_first(memory):
    memory.remember(run_id=1, goal="apollo apollo apollo", summary="about the moon")
    memory.remember(run_id=2, goal="baking bread recipes", summary="about sourdough")

    recalled = memory.recall("apollo apollo apollo", limit=2)

    assert "about the moon" in recalled[0]


def test_recall_on_empty_memory_returns_nothing(memory):
    assert memory.recall("anything") == []


def test_recall_limit_is_respected(memory):
    for i in range(5):
        memory.remember(run_id=i, goal=f"goal {i}", summary=f"summary {i}")

    assert len(memory.recall("goal", limit=2)) == 2


def test_recall_limit_above_stored_count_is_safe(memory):
    """Chroma errors if n_results exceeds the collection size unless clamped."""
    memory.remember(run_id=1, goal="only one", summary="the only entry")

    assert len(memory.recall("only one", limit=10)) == 1


def test_memory_persists_across_reopening(tmp_path, stub_embeddings):
    """A later process must see an earlier run's memory — the basis of criterion 3."""
    path = tmp_path / "chroma"
    ConversationMemory(path, embedding_function=stub_embeddings).remember(
        run_id=1, goal="the first run", summary="something worth remembering"
    )

    reopened = ConversationMemory(path, embedding_function=stub_embeddings)

    assert "something worth remembering" in reopened.recall("the first run")[0]


def test_the_goal_is_recorded_alongside_the_summary(memory):
    memory.remember(run_id=1, goal="a distinctive goal", summary="the outcome")

    assert "a distinctive goal" in memory.recall("a distinctive goal")[0]
