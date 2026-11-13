"""Exactly one outcome per task per arm.

correct                final answer matches the ground truth
wrong_answer           finished with a final answer that does not match
unrecoverable_failure  no final answer at all (sub-cause recorded)
unsupported_input      the system has no way to read the input
"""

from typing import Dict, Optional

from bench.scorer import extract_final_answer, question_scorer

CORRECT = "correct"
WRONG_ANSWER = "wrong_answer"
UNRECOVERABLE = "unrecoverable_failure"
UNSUPPORTED = "unsupported_input"
OUTCOMES = (CORRECT, WRONG_ANSWER, UNRECOVERABLE, UNSUPPORTED)


def sub_cause_for_error(error_type: str, error_message: str = "") -> str:
    """Bucket an exception into a failure sub-cause."""
    name = error_type.rsplit(".", 1)[-1]
    message = error_message.lower()
    if name == "GraphRecursionError":
        return "max_steps"
    if error_type.startswith("celery.") and name == "TimeoutError":
        return "subtask_timeout"
    if "no subtasks" in message:
        return "empty_decomposition"
    if name in ("ValidationError", "OutputParserException", "JSONDecodeError", "LengthFinishReasonError"):
        return "parse_error"
    if error_type.startswith("openai."):
        return "model_api_error"
    if error_type.startswith(("psycopg.", "redis.", "kombu.", "chromadb.")):
        return "infrastructure_error"
    return "crash"


def classify(record: Dict, ground_truth: str) -> Dict[str, Optional[str]]:
    """Label one run. ``record`` is what the harness observed, never the truth."""
    if record.get("unsupported_reason"):
        return _label(UNSUPPORTED, record["unsupported_reason"])

    # A timeout wins over any partial progress the run had written.
    if record.get("timed_out"):
        return _label(UNRECOVERABLE, "timeout")

    if record.get("error_type"):
        return _label(
            UNRECOVERABLE,
            sub_cause_for_error(record["error_type"], record.get("error_message") or ""),
        )

    # Self-repair ran out and the unattended escalation aborted the run.
    if record.get("status") == "aborted":
        return _label(UNRECOVERABLE, "retries_exhausted")

    final_text = (record.get("final_text") or "").strip()
    answer, _ = extract_final_answer(final_text)
    if not answer:
        return _label(UNRECOVERABLE, "empty_output")

    if question_scorer(answer, ground_truth):
        return _label(CORRECT, None, answer)
    return _label(WRONG_ANSWER, None, answer)


def _label(outcome: str, sub_cause: Optional[str], final_answer: Optional[str] = None) -> Dict:
    return {"outcome": outcome, "sub_cause": sub_cause, "final_answer": final_answer}
