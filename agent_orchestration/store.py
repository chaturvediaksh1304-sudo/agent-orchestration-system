"""Task history, agent configs, and escalation logs, persisted to PostgreSQL.

Every statement lives in this module. Nothing outside it imports psycopg or
knows the schema, which is what made the SQLite-to-Postgres swap a one-file
change.
"""

import json
import os
from typing import List, Optional

import psycopg
from psycopg.rows import dict_row
from pydantic import BaseModel

from agent_orchestration.state import (
    AgentSpec,
    EscalationDecision,
    RepairAttempt,
    Subtask,
    SubtaskResult,
)

DEFAULT_DSN = os.environ.get(
    "DATABASE_URL", "postgresql://orchestration:orchestration@localhost:5432/orchestration"
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id      SERIAL PRIMARY KEY,
    goal    TEXT NOT NULL,
    final   TEXT,
    user_id TEXT,
    created TIMESTAMPTZ DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS subtask_runs (
    id          SERIAL PRIMARY KEY,
    run_id      INTEGER NOT NULL REFERENCES runs(id),
    position    INTEGER NOT NULL,
    description TEXT NOT NULL,
    spec        JSONB NOT NULL,
    output      TEXT,
    ok          BOOLEAN NOT NULL DEFAULT TRUE,
    attempts    INTEGER NOT NULL DEFAULT 1,
    failures    JSONB NOT NULL DEFAULT '[]'::jsonb
);
CREATE TABLE IF NOT EXISTS escalations (
    id          SERIAL PRIMARY KEY,
    goal        TEXT NOT NULL,
    thread_id   TEXT,
    unrepaired  JSONB NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',
    action      TEXT,
    guidance    TEXT,
    user_id     TEXT,
    created     TIMESTAMPTZ DEFAULT NOW(),
    resolved_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS escalations_pending_thread
    ON escalations (thread_id) WHERE status = 'pending';
"""


class Run(BaseModel):
    """One completed orchestration run, as read back from storage."""

    id: int
    goal: str
    final: Optional[str]
    results: List[SubtaskResult]
    user_id: Optional[str] = None


class Escalation(BaseModel):
    """A run handed to a human because self-repair could not rescue it."""

    id: int
    goal: str
    thread_id: Optional[str]
    unrepaired: List[SubtaskResult]
    status: str
    action: Optional[str] = None
    guidance: Optional[str] = None
    user_id: Optional[str] = None
    resolved_at: Optional[str] = None


class Store:
    def __init__(self, dsn: Optional[str] = None):
        self.dsn = dsn or DEFAULT_DSN
        with self._connect() as conn:
            conn.execute(SCHEMA)

    def _connect(self) -> psycopg.Connection:
        return psycopg.connect(self.dsn, row_factory=dict_row, autocommit=True)

    # --- runs ------------------------------------------------------------

    def save_run(
        self,
        goal: str,
        results: List[SubtaskResult],
        final: Optional[str],
        user_id: Optional[str] = None,
    ) -> int:
        """Record a run and every subtask's agent config. Returns the run id."""
        with self._connect() as conn:
            run_id = conn.execute(
                "INSERT INTO runs (goal, final, user_id) VALUES (%s, %s, %s) RETURNING id",
                (goal, final, user_id),
            ).fetchone()["id"]
            with conn.cursor() as cursor:
                cursor.executemany(
                    "INSERT INTO subtask_runs "
                    "(run_id, position, description, spec, output, ok, attempts, failures) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                    [
                        (
                            run_id,
                            position,
                            result.subtask.description,
                            result.subtask.spec.model_dump_json(),
                            result.output,
                            result.ok,
                            result.attempts,
                            json.dumps([f.model_dump() for f in result.failures]),
                        )
                        for position, result in enumerate(results)
                    ],
                )
        return run_id

    def get_run(self, run_id: int) -> Optional[Run]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE id = %s", (run_id,)).fetchone()
            if row is None:
                return None
            subtasks = conn.execute(
                "SELECT * FROM subtask_runs WHERE run_id = %s ORDER BY position", (run_id,)
            ).fetchall()
        return self._to_run(row, subtasks)

    def load_runs(self, limit: int = 20) -> List[Run]:
        """Most recent runs first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY id DESC LIMIT %s", (limit,)
            ).fetchall()
            return [
                self._to_run(
                    row,
                    conn.execute(
                        "SELECT * FROM subtask_runs WHERE run_id = %s ORDER BY position",
                        (row["id"],),
                    ).fetchall(),
                )
                for row in rows
            ]

    # --- escalations -----------------------------------------------------
    # Logged before a human responds, so an escalation deliberately does not
    # reference runs(id): the run isn't saved until aggregate, which an
    # escalated run may never reach.

    def save_escalation(
        self,
        goal: str,
        unrepaired: List[SubtaskResult],
        thread_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> int:
        with self._connect() as conn:
            return conn.execute(
                "INSERT INTO escalations (goal, thread_id, unrepaired, user_id) "
                "VALUES (%s, %s, %s, %s) RETURNING id",
                (goal, thread_id, json.dumps([r.model_dump() for r in unrepaired]), user_id),
            ).fetchone()["id"]

    def find_pending_escalation(self, thread_id: str) -> Optional[Escalation]:
        """The open escalation for a thread, if any.

        Resuming an interrupted run re-executes the escalate node from the top,
        so the escalation must be looked up rather than written a second time.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM escalations WHERE thread_id = %s AND status = 'pending' "
                "ORDER BY id DESC LIMIT 1",
                (thread_id,),
            ).fetchone()
        return self._to_escalation(row) if row else None

    def resolve_escalation(self, escalation_id: int, decision: EscalationDecision) -> None:
        """Record the human's answer. 'abort' is kept distinct from 'resolved'."""
        status = "aborted" if decision.action == "abort" else "resolved"
        with self._connect() as conn:
            conn.execute(
                "UPDATE escalations SET status = %s, action = %s, guidance = %s, "
                "resolved_at = NOW() WHERE id = %s",
                (status, decision.action, decision.guidance, escalation_id),
            )

    def get_escalation(self, escalation_id: int) -> Optional[Escalation]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM escalations WHERE id = %s", (escalation_id,)
            ).fetchone()
        return self._to_escalation(row) if row else None

    def load_escalations(self, status: Optional[str] = None) -> List[Escalation]:
        """Most recent first, optionally filtered by status."""
        query = "SELECT * FROM escalations"
        params: tuple = ()
        if status:
            query += " WHERE status = %s"
            params = (status,)
        with self._connect() as conn:
            rows = conn.execute(query + " ORDER BY id DESC", params).fetchall()
        return [self._to_escalation(row) for row in rows]

    # --- row mapping -----------------------------------------------------

    @staticmethod
    def _to_escalation(row: dict) -> Escalation:
        return Escalation(
            id=row["id"],
            goal=row["goal"],
            thread_id=row["thread_id"],
            unrepaired=[SubtaskResult(**r) for r in row["unrepaired"]],
            status=row["status"],
            action=row["action"],
            guidance=row["guidance"],
            user_id=row["user_id"],
            resolved_at=str(row["resolved_at"]) if row["resolved_at"] else None,
        )

    @staticmethod
    def _to_run(row: dict, subtask_rows: List[dict]) -> Run:
        return Run(
            id=row["id"],
            goal=row["goal"],
            final=row["final"],
            user_id=row["user_id"],
            results=[
                SubtaskResult(
                    subtask=Subtask(
                        description=s["description"],
                        # JSONB comes back already decoded, unlike the TEXT columns
                        # this replaced.
                        spec=AgentSpec(**s["spec"]),
                    ),
                    output=s["output"],
                    ok=s["ok"],
                    attempts=s["attempts"],
                    failures=[RepairAttempt(**f) for f in s["failures"]],
                )
                for s in subtask_rows
            ],
        )
