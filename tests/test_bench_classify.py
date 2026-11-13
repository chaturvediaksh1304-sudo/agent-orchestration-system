"""Each run gets exactly one label, and it must be the right one.

The first group drives the real subprocess path with stand-in child scripts, so
a crash, a timeout, a wrong answer and a correct answer are simulated the same
way the benchmark meets them. The second group pins the label rules directly.
"""

import json
import sys
import textwrap

import pytest

from bench.classify import (
    CORRECT,
    OUTCOMES,
    UNRECOVERABLE,
    UNSUPPORTED,
    WRONG_ANSWER,
    classify,
    sub_cause_for_error,
)
from bench.runner import run_child


def _child(tmp_path, body, timeout_s=20):
    """Run a stand-in for the agent process and return the harness record."""
    out = tmp_path / "out.json"
    script = "import json, sys, time\nOUT = sys.argv[1]\n" + textwrap.dedent(body)
    return run_child([sys.executable, "-c", script, str(out)], out, timeout_s)


def _report(final_text):
    return f"json.dump({{'finished': True, 'final_text': {final_text!r}}}, open(OUT, 'w'))"


def test_a_correct_answer_is_labelled_correct(tmp_path):
    record = _child(tmp_path, _report("Thinking...\nFINAL ANSWER: 1,234"))

    label = classify(record, "1234")

    assert label["outcome"] == CORRECT
    assert label["final_answer"] == "1,234"


def test_a_wrong_answer_is_labelled_wrong_answer(tmp_path):
    record = _child(tmp_path, _report("FINAL ANSWER: London"))

    label = classify(record, "Paris")

    assert label["outcome"] == WRONG_ANSWER
    assert label["sub_cause"] is None
    assert label["final_answer"] == "London"


def test_a_crash_is_an_unrecoverable_failure(tmp_path):
    record = _child(tmp_path, "raise RuntimeError('agent blew up')")

    label = classify(record, "Paris")

    assert record["error_type"] == "ChildProcessDied"
    assert (label["outcome"], label["sub_cause"]) == (UNRECOVERABLE, "crash")


def test_a_reported_exception_keeps_its_type(tmp_path):
    record = _child(
        tmp_path,
        """
        json.dump({'finished': False, 'error_type': 'langgraph.errors.GraphRecursionError',
                   'error_message': 'Recursion limit of 25 reached'}, open(OUT, 'w'))
        """,
    )

    label = classify(record, "Paris")

    assert (label["outcome"], label["sub_cause"]) == (UNRECOVERABLE, "max_steps")


def test_a_timeout_is_an_unrecoverable_failure(tmp_path):
    record = _child(tmp_path, "time.sleep(60)", timeout_s=1)

    label = classify(record, "Paris")

    assert record["timed_out"] is True
    assert record["latency_s"] < 10
    assert (label["outcome"], label["sub_cause"]) == (UNRECOVERABLE, "timeout")


def test_a_timeout_wins_over_partial_progress(tmp_path):
    """A run killed mid-flight must not be scored on whatever it had so far."""
    record = _child(
        tmp_path,
        _report("FINAL ANSWER: Paris").replace("'finished': True", "'finished': False")
        + "\ntime.sleep(60)",
        timeout_s=1,
    )

    assert classify(record, "Paris")["sub_cause"] == "timeout"


def test_exhausted_retries_are_an_unrecoverable_failure():
    label = classify({"finished": True, "status": "aborted", "final_text": None}, "Paris")

    assert (label["outcome"], label["sub_cause"]) == (UNRECOVERABLE, "retries_exhausted")


@pytest.mark.parametrize("final_text", [None, "", "   ", "FINAL ANSWER:"])
def test_finishing_with_nothing_to_say_is_unrecoverable(final_text):
    label = classify({"finished": True, "final_text": final_text}, "Paris")

    assert (label["outcome"], label["sub_cause"]) == (UNRECOVERABLE, "empty_output")


def test_unsupported_input_is_never_counted_as_an_agent_failure():
    label = classify({"unsupported_reason": "attachment_not_text:xlsx"}, "Paris")

    assert (label["outcome"], label["sub_cause"]) == (UNSUPPORTED, "attachment_not_text:xlsx")


def test_an_answer_without_the_marker_is_still_scored():
    assert classify({"finished": True, "final_text": "Paris"}, "Paris")["outcome"] == CORRECT
    assert classify({"finished": True, "final_text": "It is Paris."}, "Paris")["outcome"] == WRONG_ANSWER


@pytest.mark.parametrize(
    "error_type, message, expected",
    [
        ("langgraph.errors.GraphRecursionError", "", "max_steps"),
        ("celery.exceptions.TimeoutError", "The operation timed out.", "subtask_timeout"),
        ("builtins.ValueError", "The supervisor returned no subtasks for goal: 'x'", "empty_decomposition"),
        ("pydantic_core._pydantic_core.ValidationError", "1 validation error", "parse_error"),
        ("langchain_core.exceptions.OutputParserException", "bad json", "parse_error"),
        ("openai.APIConnectionError", "Connection error.", "model_api_error"),
        ("psycopg.OperationalError", "connection refused", "infrastructure_error"),
        ("builtins.KeyError", "'final'", "crash"),
        ("ChildProcessDied", "exited with code -11", "crash"),
    ],
)
def test_errors_are_bucketed_by_cause(error_type, message, expected):
    assert sub_cause_for_error(error_type, message) == expected


def test_every_label_is_one_of_the_four_outcomes():
    records = [
        {"unsupported_reason": "x"},
        {"timed_out": True},
        {"error_type": "builtins.RuntimeError"},
        {"finished": True, "status": "aborted"},
        {"finished": True, "final_text": "FINAL ANSWER: Paris"},
        {"finished": True, "final_text": "FINAL ANSWER: Rome"},
    ]

    assert all(classify(r, "Paris")["outcome"] in OUTCOMES for r in records)
