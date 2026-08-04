import io

from agent_orchestration.escalation import (
    abort_handler,
    describe,
    interactive_handler,
)
from agent_orchestration.state import AgentSpec, RepairAttempt, Subtask, SubtaskResult

UNREPAIRED = [
    SubtaskResult(
        subtask=Subtask(
            description="write the report",
            spec=AgentSpec(role="writer", system_prompt="p", tools=["write_file"]),
        ),
        output="",
        ok=False,
        attempts=3,
        failures=[RepairAttempt(error="RuntimeError: disk full", diagnosis="no space left")],
    )
]


def _interactive(text):
    out = io.StringIO()
    decision = interactive_handler("the goal", UNREPAIRED, io.StringIO(text), out)
    return decision, out.getvalue()


def test_description_carries_what_a_human_needs():
    rendered = describe("the goal", UNREPAIRED)

    assert "the goal" in rendered
    assert "write the report" in rendered
    assert "disk full" in rendered
    assert "no space left" in rendered
    assert "3 attempt(s)" in rendered


def test_unattended_runs_abort_rather_than_hang():
    """Automation must never block on a prompt nobody will answer."""
    assert abort_handler("the goal", UNREPAIRED).action == "abort"


def test_operator_can_retry_with_guidance():
    decision, _ = _interactive("retry\nuse read_file instead\n")

    assert decision.action == "retry"
    assert decision.guidance == "use read_file instead"


def test_operator_can_skip():
    decision, _ = _interactive("skip\n")

    assert decision.action == "skip"
    assert decision.guidance == ""


def test_operator_can_abort():
    assert _interactive("abort\n")[0].action == "abort"


def test_invalid_input_reprompts_rather_than_crashing():
    decision, output = _interactive("banana\nskip\n")

    assert decision.action == "skip"
    assert "Choose one of" in output


def test_closed_input_aborts_instead_of_looping_forever():
    """A closed stdin must not spin the re-prompt loop."""
    assert _interactive("")[0].action == "abort"


def test_the_failure_detail_is_shown_to_the_operator():
    _, output = _interactive("abort\n")

    assert "disk full" in output
