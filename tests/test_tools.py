"""Tool behaviour, and the workspace boundary that contains it.

Paths reaching these tools are written by a model, so the boundary is a trust
boundary: the escape cases below matter more than the happy path.
"""

from pathlib import Path

import pytest

from agent_orchestration.tools import build_tools


@pytest.fixture
def tools(tmp_path):
    return build_tools(tmp_path)


def test_write_then_read_round_trip(tools):
    tools["write_file"].invoke({"path": "notes.txt", "content": "hello"})

    assert tools["read_file"].invoke({"path": "notes.txt"}) == "hello"


def test_write_creates_parent_directories(tools):
    tools["write_file"].invoke({"path": "a/b/deep.txt", "content": "x"})

    assert tools["read_file"].invoke({"path": "a/b/deep.txt"}) == "x"


def test_list_files_reports_written_files(tools):
    tools["write_file"].invoke({"path": "one.txt", "content": "1"})
    tools["write_file"].invoke({"path": "nested/two.txt", "content": "2"})

    listed = tools["list_files"].invoke({})

    assert "one.txt" in listed
    assert "nested/two.txt" in listed


def test_list_files_on_empty_workspace(tools):
    assert tools["list_files"].invoke({}) == ""


def test_read_missing_file_raises_clearly(tools):
    with pytest.raises(FileNotFoundError):
        tools["read_file"].invoke({"path": "nope.txt"})


@pytest.mark.parametrize(
    "escape",
    [
        "../outside.txt",
        "nested/../../outside.txt",
        "/etc/passwd",
    ],
)
def test_paths_outside_workspace_are_rejected(tools, escape):
    with pytest.raises(ValueError, match="outside the workspace"):
        tools["read_file"].invoke({"path": escape})

    with pytest.raises(ValueError, match="outside the workspace"):
        tools["write_file"].invoke({"path": escape, "content": "pwned"})


def test_tilde_stays_literal_and_never_reaches_home(tmp_path):
    """A "~" from the model is a directory name, not the user's home directory."""
    workspace = tmp_path / "workspace"
    tools = build_tools(workspace)

    tools["write_file"].invoke({"path": "~/secrets.txt", "content": "contained"})

    assert (workspace / "~" / "secrets.txt").read_text() == "contained"
    assert not (Path.home() / "secrets.txt").exists()


def test_symlink_escape_is_rejected(tmp_path):
    """A symlink inside the workspace must not become a way out of it."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("classified")
    (workspace / "backdoor").symlink_to(outside)

    tools = build_tools(workspace)

    with pytest.raises(ValueError, match="outside the workspace"):
        tools["read_file"].invoke({"path": "backdoor/secret.txt"})


def test_workspace_is_created_if_absent(tmp_path):
    build_tools(tmp_path / "fresh")

    assert (tmp_path / "fresh").is_dir()


class _FakeSearch:
    """Stands in for the search client so the default suite stays offline."""

    results = [{"title": "Apollo 11", "href": "https://example.test/apollo", "body": "Landed in 1969."}]

    def text(self, query, max_results):
        return self.results[:max_results]


def test_web_search_returns_title_url_and_snippet(tools, monkeypatch):
    import ddgs

    monkeypatch.setattr(ddgs, "DDGS", _FakeSearch)

    found = tools["web_search"].invoke({"query": "apollo 11"})

    assert "Apollo 11" in found
    assert "https://example.test/apollo" in found
    assert "Landed in 1969." in found


def test_web_search_with_no_results_says_so(tools, monkeypatch):
    import ddgs

    monkeypatch.setattr(ddgs, "DDGS", type("Empty", (_FakeSearch,), {"results": []}))

    assert tools["web_search"].invoke({"query": "nothing matches"}) == "No results."
