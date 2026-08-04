import pytest
from langchain_core.messages import AIMessage

from agent_orchestration import cli


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])

    assert exc.value.code == 0
    assert "goal" in capsys.readouterr().out


def test_run_prints_the_aggregated_result(
    stub_model, two_subtasks, tmp_path, capsys, monkeypatch
):
    llm = stub_model(
        [two_subtasks, AIMessage(content="a"), AIMessage(content="b"), AIMessage(content="THE ANSWER")]
    )
    monkeypatch.setattr(cli, "build_llm", lambda provider, model: llm)

    exit_code = cli.main(["do the thing", "--workspace", str(tmp_path), "--no-memory"])

    assert exit_code == 0
    assert "THE ANSWER" in capsys.readouterr().out


def test_no_memory_writes_nothing_to_disk(stub_model, two_subtasks, tmp_path, monkeypatch):
    """--no-memory must not create the history database or the Chroma directory."""
    llm = stub_model(
        [two_subtasks, AIMessage(content="a"), AIMessage(content="b"), AIMessage(content="done")]
    )
    monkeypatch.setattr(cli, "build_llm", lambda provider, model: llm)
    monkeypatch.chdir(tmp_path)

    cli.main(["a goal", "--workspace", str(tmp_path / "ws"), "--no-memory"])

    assert not (tmp_path / ".orchestration").exists()


@pytest.mark.integration
def test_the_database_url_is_honoured(stub_model, two_subtasks, tmp_path, monkeypatch, store):
    """--database-url must reach the store, not just the default DSN."""
    llm = stub_model(
        [two_subtasks, AIMessage(content="a"), AIMessage(content="b"), AIMessage(content="done")]
    )
    monkeypatch.setattr(cli, "build_llm", lambda provider, model: llm)
    monkeypatch.setattr(cli, "ConversationMemory", lambda path: None)

    cli.main(
        [
            "a goal",
            "--workspace", str(tmp_path / "ws"),
            "--database-url", store.dsn,
        ]
    )

    assert store.load_runs()[0].goal == "a goal"


def test_durable_without_a_store_is_rejected(tmp_path, capsys):
    """Otherwise the run halts with no escalation id and can never be resumed."""
    exit_code = cli.main(["a goal", "--durable", "--no-memory", "--workspace", str(tmp_path)])

    assert exit_code == 1
    assert "--no-memory" in capsys.readouterr().err


def test_unknown_provider_fails_with_a_clear_message(tmp_path, capsys):
    """Reported before the database is opened, so it needs no Postgres."""
    exit_code = cli.main(["a goal", "--provider", "gemini", "--workspace", str(tmp_path)])

    assert exit_code == 1
    assert "gemini" in capsys.readouterr().err


