"""Conversation memory, persisted to ChromaDB.

What a run was trying to do and what came of it, stored so a later run can
retrieve the relevant parts by similarity rather than by exact key.
"""

from pathlib import Path
from typing import List, Optional, Union

import chromadb

COLLECTION = "conversation_memory"


class ConversationMemory:
    def __init__(self, path: Union[str, Path], embedding_function: Optional[object] = None):
        """``embedding_function`` defaults to Chroma's local model.

        Tests pass a deterministic stub so the suite stays offline and free; the
        default downloads an ONNX model on first use.
        """
        client = chromadb.PersistentClient(path=str(path))
        kwargs = {"embedding_function": embedding_function} if embedding_function else {}
        self.collection = client.get_or_create_collection(COLLECTION, **kwargs)

    def remember(self, run_id: int, goal: str, summary: str) -> None:
        """Store one run's goal and outcome as a single retrievable document."""
        self.collection.upsert(
            ids=[str(run_id)],
            documents=[f"Goal: {goal}\nOutcome: {summary}"],
            metadatas=[{"run_id": run_id, "goal": goal}],
        )

    def recall(self, goal: str, limit: int = 3) -> List[str]:
        """Return prior run documents most similar to ``goal``, closest first."""
        available = self.collection.count()
        if not available:
            return []
        # Chroma raises if n_results exceeds what the collection holds.
        result = self.collection.query(query_texts=[goal], n_results=min(limit, available))
        return result["documents"][0]
